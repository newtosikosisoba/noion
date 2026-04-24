#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""music_theory.py — 音楽理論補正レイヤー

変更理由: AI出力 + 音楽理論補正 で人間の耳コピレベルを目指す。
mimikopi.py を壊さず、ノートリストに対して独立して適用できる補正群。

対応機能:
  ① estimate_chords        : Chroma からコード推定 (maj/min/7th)
  ② correct_by_chord       : コード補正 (同時発音抑制 / 外れ音削除 / コードトーン優先)
  ③ estimate_key + filter_by_scale : キー推定 + スケール外削除
  ④ stabilize_bass         : ベース領域の安定化
  ⑤ quantize_rhythm        : 16分音符グリッド量子化
  ⑥ score_confidence       : ノート信頼度スコア
  ⑦ prioritize_harmony     : 和音優先・不自然な配置修正

依存: numpy, librosa (既に mimikopi.py でインストール済み)

使い方:
    from music_theory import correct_notes
    refined = correct_notes(notes, audio=y, sr=44100, tempo=120.0,
                            beat_times=beats)
"""

import logging
from collections import Counter

import numpy as np

log = logging.getLogger("music_theory")

# =====================================================
# 定数・テンプレート
# =====================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# コードテンプレート (12次元 chroma 重みベクトル)
# 値は各ピッチクラスの期待強度。Root基準で表記。
CHORD_TEMPLATES = {
    'maj':  [1.0, 0, 0, 0, 0.8, 0, 0, 0.9, 0, 0, 0, 0],
    'min':  [1.0, 0, 0, 0.8, 0, 0, 0, 0.9, 0, 0, 0, 0],
    'maj7': [1.0, 0, 0, 0, 0.8, 0, 0, 0.9, 0, 0, 0, 0.6],
    'min7': [1.0, 0, 0, 0.8, 0, 0, 0, 0.9, 0, 0, 0.6, 0],
    'dom7': [1.0, 0, 0, 0, 0.8, 0, 0, 0.9, 0, 0, 0.6, 0],
    'sus4': [1.0, 0, 0, 0, 0, 0.8, 0, 0.9, 0, 0, 0, 0],
    'dim':  [1.0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 0],
    'aug':  [1.0, 0, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0],
}

# コード構成音のインターバル (root からの半音数)
CHORD_INTERVALS = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
    'sus4': [0, 5, 7],
    'dim':  [0, 3, 6],
    'aug':  [0, 4, 8],
}

# Krumhansl-Schmuckler 調性プロファイル
KEY_PROFILE_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KEY_PROFILE_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# メジャー / ナチュラルマイナースケールのピッチクラス集合
SCALE_MAJOR = {0, 2, 4, 5, 7, 9, 11}
SCALE_MINOR = {0, 2, 3, 5, 7, 8, 10}  # ナチュラルマイナー
# 和声的マイナー (7thを上げたもの) を併せて許容
SCALE_MINOR_HARMONIC = {0, 2, 3, 5, 7, 8, 11}


# =====================================================
# ① コード推定
# =====================================================

def estimate_chords(audio, sr=44100, beat_times=None, hop_length=512):
    """音声とビート時刻からコード列を推定する。

    Args:
        audio      : 1D ndarray (モノラル音声)
        sr         : サンプリングレート
        beat_times : ビート時刻列 (秒). None の場合は 0.5 秒グリッドで等分
        hop_length : Chroma 抽出の hop

    Returns:
        [(onset_s, dur_s, root_pc, quality, score), ...]
            root_pc : 0-11 (C=0)
            quality : 'maj' / 'min' / 'maj7' ...
            score   : テンプレートマッチスコア 0-1
    """
    try:
        import librosa
    except ImportError:
        log.error("estimate_chords: librosa が必要です")
        return []

    if audio is None or len(audio) == 0:
        return []

    # ハーモニック成分のみ使用（ドラム除外でコード推定を安定化）
    try:
        y_h = librosa.effects.harmonic(audio)
    except Exception:
        y_h = audio

    chroma = librosa.feature.chroma_cqt(
        y=y_h, sr=sr, hop_length=hop_length
    )  # (12, T_frames)
    T_frames = chroma.shape[1]
    total_dur = len(audio) / sr

    # ビート時刻を整備
    if beat_times is None or len(beat_times) == 0:
        # 0.5 秒ごとに等分
        step = 0.5
        beat_times = np.arange(0, total_dur, step)
    beat_times = np.asarray(beat_times, dtype=float)
    beat_frames = librosa.time_to_frames(
        beat_times, sr=sr, hop_length=hop_length
    )

    events = []
    for i, bf in enumerate(beat_frames):
        if bf >= T_frames:
            break
        end_f = beat_frames[i + 1] if i + 1 < len(beat_frames) else T_frames
        end_f = min(max(int(end_f), int(bf) + 1), T_frames)
        segment = chroma[:, int(bf):end_f]
        if segment.size == 0:
            continue
        cv = np.mean(segment, axis=1)
        norm = np.linalg.norm(cv) + 1e-9
        cv = cv / norm

        best_score = -1.0
        best_root = 0
        best_q = 'maj'
        for root in range(12):
            for q, tmpl in CHORD_TEMPLATES.items():
                t = np.array(tmpl)
                t = t / (np.linalg.norm(t) + 1e-9)
                rolled = np.roll(t, root)
                s = float(np.dot(cv, rolled))
                if s > best_score:
                    best_score = s
                    best_root = root
                    best_q = q

        onset = beat_times[i]
        next_t = (beat_times[i + 1] if i + 1 < len(beat_times)
                  else onset + 0.5)
        dur = max(next_t - onset, 0.05)
        events.append((onset, dur, best_root, best_q, best_score))

    # 低信頼(スコア < 0.35)かつ前後が同じコードなら結合
    events = _merge_adjacent_chords(events)
    log.info("estimate_chords: %d events", len(events))
    return events


def _merge_adjacent_chords(events):
    """隣接する同コードをマージ"""
    if not events:
        return events
    merged = [events[0]]
    for ev in events[1:]:
        onset, dur, root, q, score = ev
        p_onset, p_dur, p_root, p_q, p_score = merged[-1]
        if p_root == root and p_q == q:
            merged[-1] = (p_onset, onset + dur - p_onset, p_root, p_q,
                          max(p_score, score))
        else:
            merged.append(ev)
    return merged


# =====================================================
# ③ キー推定
# =====================================================

def estimate_key(audio, sr=44100, hop_length=512):
    """Krumhansl-Schmuckler 法でキーを推定する。

    Returns:
        (key_root, key_mode, score)
            key_root : 0-11 (C=0)
            key_mode : 'major' / 'minor'
            score    : 相関スコア (高いほど確信度大)
    """
    try:
        import librosa
    except ImportError:
        log.error("estimate_key: librosa が必要です")
        return (0, 'major', 0.0)

    if audio is None or len(audio) == 0:
        return (0, 'major', 0.0)

    try:
        y_h = librosa.effects.harmonic(audio)
    except Exception:
        y_h = audio

    chroma = librosa.feature.chroma_cqt(
        y=y_h, sr=sr, hop_length=hop_length
    )
    chroma_sum = np.sum(chroma, axis=1)
    norm = np.sum(chroma_sum) + 1e-9
    chroma_sum = chroma_sum / norm

    best_score = -np.inf
    best_root = 0
    best_mode = 'major'
    for root in range(12):
        maj = np.roll(KEY_PROFILE_MAJOR, root)
        mn = np.roll(KEY_PROFILE_MINOR, root)
        # Pearson相関
        for profile, mode in ((maj, 'major'), (mn, 'minor')):
            p = profile / (np.sum(profile) + 1e-9)
            s = float(np.corrcoef(chroma_sum, p)[0, 1])
            if s > best_score:
                best_score = s
                best_root = root
                best_mode = mode

    log.info("estimate_key: %s %s (score=%.3f)",
             NOTE_NAMES[best_root], best_mode, best_score)
    return (best_root, best_mode, best_score)


def _scale_pc_set(key_root, key_mode):
    """キーから許容ピッチクラス集合を返す"""
    if key_mode == 'minor':
        base = SCALE_MINOR | SCALE_MINOR_HARMONIC
    else:
        base = SCALE_MAJOR
    return {(p + key_root) % 12 for p in base}


def filter_by_scale(notes, key_root, key_mode, tolerance=0.15):
    """スケール外のノートを削除。
    ただし短いノート(経過音)は tolerance 以下の割合まで許容。

    Args:
        notes      : [(onset, dur, midi, vel), ...]
        key_root   : 0-11
        key_mode   : 'major' / 'minor'
        tolerance  : スケール外を許容する音の最大デュレーション(秒)
    """
    if not notes:
        return notes
    allowed = _scale_pc_set(key_root, key_mode)
    out = []
    removed = 0
    for (t, d, m, v) in notes:
        pc = int(m) % 12
        if pc in allowed:
            out.append((t, d, m, v))
        elif d <= tolerance:
            # 経過音として許容 (ただしベロシティ減衰)
            out.append((t, d, m, max(1, int(v * 0.7))))
        else:
            removed += 1
    log.debug("filter_by_scale: removed %d notes", removed)
    return out


# =====================================================
# ② コード補正 + ⑦ 和音優先
# =====================================================

def _chord_pitch_classes(root_pc, quality):
    """コード構成音のピッチクラス集合"""
    ivs = CHORD_INTERVALS.get(quality, CHORD_INTERVALS['maj'])
    return {(root_pc + iv) % 12 for iv in ivs}


def _active_chord_at(chords, t):
    """時刻 t に対応するコードを返す。なければ None"""
    for (onset, dur, root, q, score) in chords:
        if onset <= t < onset + dur:
            return (root, q, score)
    # 最も近い前のコードを使う (フォールバック)
    prev = None
    for (onset, dur, root, q, score) in chords:
        if onset <= t:
            prev = (root, q, score)
        else:
            break
    return prev


def correct_by_chord(notes, chords, max_polyphony=4,
                     foreign_velocity_penalty=0.5,
                     remove_foreign_short=True):
    """AIノート出力をコードで補正する。

    処理:
      1. 各ノート時刻のコードを取得
      2. コードトーン以外のノートは:
         - 短い場合 (<0.08s) → 削除
         - それ以外 → ベロシティを減衰 (経過音として残す)
      3. 同時発音数が max_polyphony を超える場合、
         コードトーン優先 + ベロシティ上位を残す

    Args:
        notes                     : ノートリスト
        chords                    : estimate_chords() の出力
        max_polyphony             : 同時発音上限
        foreign_velocity_penalty  : コード外音のベロシティ係数
        remove_foreign_short      : 短いコード外音を削除するか

    Returns:
        補正後のノートリスト
    """
    if not notes:
        return notes
    if not chords:
        return notes

    # 1. コード適合補正
    filtered = []
    removed = 0
    for (t, d, m, v) in notes:
        active = _active_chord_at(chords, t)
        if active is None:
            filtered.append((t, d, m, v))
            continue
        root, q, _ = active
        chord_pcs = _chord_pitch_classes(root, q)
        pc = int(m) % 12
        if pc in chord_pcs:
            filtered.append((t, d, m, v))
        else:
            if remove_foreign_short and d < 0.08:
                removed += 1
                continue
            filtered.append((t, d, m, max(1, int(v * foreign_velocity_penalty))))
    log.debug("correct_by_chord: removed %d foreign-short notes", removed)

    # 2. 同時発音数の制限 (0.05秒窓で判定)
    filtered.sort(key=lambda n: n[0])
    result = []
    window = 0.05
    i = 0
    n = len(filtered)
    while i < n:
        t_i = filtered[i][0]
        group = [filtered[i]]
        j = i + 1
        while j < n and filtered[j][0] - t_i < window:
            group.append(filtered[j])
            j += 1
        if len(group) <= max_polyphony:
            result.extend(group)
        else:
            active = _active_chord_at(chords, t_i)
            chord_pcs = (_chord_pitch_classes(*active[:2])
                         if active else set())
            # コードトーン → ベロシティ順で優先
            scored = [
                (0 if (int(m) % 12) in chord_pcs else 1, -v, note)
                for note in group
                for (_, _, m, v) in [note]
            ]
            scored.sort(key=lambda x: (x[0], x[1]))
            result.extend([s[2] for s in scored[:max_polyphony]])
        i = j

    return sorted(result, key=lambda n: n[0])


def prioritize_harmony(notes, chords, solo_midi_range=None):
    """単音ではなく和音を優先的に残すロジック。

    短時間内に複数ピッチが鳴っていれば和音として扱い、
    孤立した単音でコード外の場合は削除候補。

    Args:
        notes             : ノートリスト
        chords            : コード列
        solo_midi_range   : メロディとして常に残す MIDI 範囲 (例: (60, 96))
    """
    if not notes or not chords:
        return notes

    if solo_midi_range is None:
        solo_midi_range = (60, 127)  # C4以上はメロディ扱い

    notes_sorted = sorted(notes, key=lambda n: n[0])
    result = []
    n = len(notes_sorted)
    window = 0.08

    for i, (t, d, m, v) in enumerate(notes_sorted):
        # ソロ(メロディ)領域は常に残す
        if solo_midi_range[0] <= m <= solo_midi_range[1]:
            result.append((t, d, m, v))
            continue

        # 近傍に他ノートがあるか確認
        near_count = 0
        for j in range(max(0, i - 4), min(n, i + 5)):
            if j == i:
                continue
            if abs(notes_sorted[j][0] - t) < window:
                near_count += 1

        active = _active_chord_at(chords, t)
        is_chord_tone = False
        if active:
            is_chord_tone = (int(m) % 12) in _chord_pitch_classes(*active[:2])

        # 単音で、コード外、かつメロディ領域外 → 削除
        if near_count == 0 and not is_chord_tone:
            continue
        result.append((t, d, m, v))

    return result


# =====================================================
# ④ ベース専用処理
# =====================================================

def stabilize_bass(notes, bass_range=(24, 55),
                   min_dur=0.08, merge_gap=0.06):
    """ベース領域のノートを安定化する。

    処理:
      1. ベース領域内で同ピッチ連続ノートを統合
      2. 短すぎるベースノート (< min_dur) を削除 or 次ノートに統合
      3. 同時刻に複数ベース音がある場合は最低音のみ残す

    Args:
        notes      : ノートリスト
        bass_range : ベース MIDI 範囲 (inclusive)
        min_dur    : 最小デュレーション(秒)
        merge_gap  : このギャップ以下なら統合
    """
    if not notes:
        return notes

    bass = []
    other = []
    for n in notes:
        if bass_range[0] <= n[2] <= bass_range[1]:
            bass.append(n)
        else:
            other.append(n)

    if not bass:
        return notes

    bass.sort(key=lambda n: n[0])

    # 1. 同ピッチ連続ノート統合
    merged = []
    for note in bass:
        if not merged:
            merged.append(list(note))
            continue
        last = merged[-1]
        gap = note[0] - (last[0] + last[1])
        if note[2] == last[2] and gap <= merge_gap:
            last[1] = note[0] + note[1] - last[0]
            last[3] = max(last[3], note[3])
        else:
            merged.append(list(note))

    # 2. 短すぎるノート削除
    merged = [tuple(m) for m in merged if m[1] >= min_dur]

    # 3. 同時刻の最低音のみ残す (0.04秒窓)
    result = []
    i = 0
    n = len(merged)
    window = 0.04
    while i < n:
        t_i = merged[i][0]
        group = [merged[i]]
        j = i + 1
        while j < n and merged[j][0] - t_i < window:
            group.append(merged[j])
            j += 1
        lowest = min(group, key=lambda x: x[2])
        result.append(lowest)
        i = j

    log.debug("stabilize_bass: %d → %d", len(bass), len(result))
    return sorted(result + other, key=lambda n: n[0])


# =====================================================
# ⑤ リズム補正 (16分音符グリッド量子化)
# =====================================================

def quantize_rhythm(notes, tempo=120.0, beat_times=None,
                    subdivision=16, strength=0.7):
    """ノートを subdivision 分音符グリッドに量子化する。

    Args:
        notes       : ノートリスト
        tempo       : BPM
        beat_times  : ビート時刻列 (あれば優先使用。なければ tempo から生成)
        subdivision : 16 = 16分音符, 8 = 8分音符
        strength    : 0-1 (1.0=完全量子化, 0.0=無変更)
    """
    if not notes:
        return notes
    if strength <= 0:
        return notes

    tempo = max(float(tempo), 30.0)
    beat_dur = 60.0 / tempo
    step = beat_dur * 4.0 / subdivision  # 16 なら 1/4 拍

    if beat_times is not None and len(beat_times) >= 2:
        beat_times = np.asarray(beat_times, dtype=float)
        # ビート間を subdivision に細分
        grid_points = []
        for i in range(len(beat_times) - 1):
            b0 = beat_times[i]
            b1 = beat_times[i + 1]
            n_div = max(1, subdivision // 4)
            for k in range(n_div):
                grid_points.append(b0 + (b1 - b0) * k / n_div)
        grid_points.append(beat_times[-1])
        grid = np.array(sorted(set(grid_points)))
    else:
        total = max(n[0] + n[1] for n in notes)
        grid = np.arange(0, total + step, step)

    out = []
    for (t, d, m, v) in notes:
        # 最近傍グリッド点を検索
        idx = int(np.argmin(np.abs(grid - t)))
        t_grid = float(grid[idx])
        delta = t_grid - t
        # 強度に応じてスナップ (大きく離れている場合は控えめに)
        if abs(delta) < step * 0.5:
            new_t = t + delta * strength
        else:
            new_t = t
        # デュレーションも近いグリッドにスナップ
        end = t + d
        idx_e = int(np.argmin(np.abs(grid - end)))
        end_grid = float(grid[idx_e])
        if abs(end_grid - end) < step * 0.5:
            new_end = end + (end_grid - end) * strength
        else:
            new_end = end
        new_d = max(new_end - new_t, 0.02)
        out.append((max(0.0, new_t), new_d, m, v))

    return sorted(out, key=lambda n: n[0])


# =====================================================
# ⑥ ノート信頼度スコア
# =====================================================

def score_confidence(notes, chords=None, key_root=None, key_mode=None):
    """各ノートに信頼度スコア (0-1) を付与する。

    スコア要素:
      + コードトーンか (最大 +0.35)
      + スケール内か (最大 +0.25)
      + ベロシティ (最大 +0.20)
      + デュレーション妥当性 (最大 +0.20)

    Returns:
        [(onset, dur, midi, vel, confidence), ...]
    """
    if not notes:
        return []

    scored = []
    scale_pcs = None
    if key_root is not None and key_mode is not None:
        scale_pcs = _scale_pc_set(key_root, key_mode)

    for (t, d, m, v) in notes:
        pc = int(m) % 12
        score = 0.0

        # コード適合
        if chords:
            active = _active_chord_at(chords, t)
            if active:
                if pc in _chord_pitch_classes(*active[:2]):
                    score += 0.35
                else:
                    score += 0.05  # コード外でも完全ゼロにはしない
            else:
                score += 0.15

        # スケール適合
        if scale_pcs is not None:
            if pc in scale_pcs:
                score += 0.25
            else:
                score += 0.05

        # ベロシティ (30-120 を 0-0.2 にマップ)
        score += max(0.0, min(0.20, (v - 30) / 450.0))

        # デュレーション妥当性 (0.05s - 2.0s が最も信頼)
        if 0.05 <= d <= 2.0:
            score += 0.20
        elif 0.02 <= d < 0.05 or 2.0 < d <= 4.0:
            score += 0.10

        score = float(min(1.0, max(0.0, score)))
        scored.append((t, d, m, v, score))

    return scored


def filter_by_confidence(scored_notes, threshold=0.35):
    """信頼度スコア付きノートから低信頼を除外し、
    標準形 (t,d,m,v) に戻す。
    """
    out = []
    removed = 0
    for rec in scored_notes:
        if len(rec) == 5:
            t, d, m, v, c = rec
            if c >= threshold:
                out.append((t, d, m, v))
            else:
                removed += 1
        else:
            out.append(rec)
    log.debug("filter_by_confidence: removed %d low-conf notes", removed)
    return out


# =====================================================
# パイプライン統合
# =====================================================

class MusicTheoryCorrector:
    """7機能を統合した補正クラス。

    使い方:
        mtc = MusicTheoryCorrector(audio=y, sr=44100,
                                   tempo=120, beat_times=beats)
        refined = mtc.correct(notes)
    """

    def __init__(self, audio=None, sr=44100, tempo=120.0,
                 beat_times=None, hop_length=512):
        self.audio = audio
        self.sr = sr
        self.tempo = float(tempo)
        self.beat_times = beat_times
        self.hop_length = hop_length
        self.chords = []
        self.key_root = 0
        self.key_mode = 'major'
        self.key_score = 0.0

        if audio is not None and len(audio) > 0:
            self.chords = estimate_chords(
                audio, sr=sr, beat_times=beat_times,
                hop_length=hop_length,
            )
            self.key_root, self.key_mode, self.key_score = estimate_key(
                audio, sr=sr, hop_length=hop_length
            )

    def correct(self, notes,
                apply_scale=True,
                apply_chord=True,
                apply_bass=True,
                apply_rhythm=True,
                apply_confidence=True,
                apply_harmony=True,
                confidence_threshold=0.30,
                max_polyphony=4,
                quantize_strength=0.6):
        """フルパイプラインで補正を適用"""
        if not notes:
            return notes

        log.info("MusicTheoryCorrector.correct: input=%d notes", len(notes))

        # ④ ベース安定化
        if apply_bass:
            notes = stabilize_bass(notes)

        # ③ スケールフィルタ
        if apply_scale and self.key_score > 0.3:
            notes = filter_by_scale(notes, self.key_root, self.key_mode)

        # ② コード補正
        if apply_chord and self.chords:
            notes = correct_by_chord(
                notes, self.chords, max_polyphony=max_polyphony
            )

        # ⑦ 和音優先
        if apply_harmony and self.chords:
            notes = prioritize_harmony(notes, self.chords)

        # ⑤ リズム量子化
        if apply_rhythm:
            notes = quantize_rhythm(
                notes, tempo=self.tempo,
                beat_times=self.beat_times,
                strength=quantize_strength,
            )

        # ⑥ 信頼度スコアで低信頼を削除
        if apply_confidence:
            scored = score_confidence(
                notes, chords=self.chords,
                key_root=self.key_root, key_mode=self.key_mode,
            )
            notes = filter_by_confidence(scored, threshold=confidence_threshold)

        log.info("MusicTheoryCorrector.correct: output=%d notes", len(notes))
        return sorted(notes, key=lambda n: n[0])


def correct_notes(notes, audio=None, sr=44100, tempo=120.0,
                  beat_times=None, **kwargs):
    """ワンショット関数。MusicTheoryCorrector のショートカット。"""
    mtc = MusicTheoryCorrector(
        audio=audio, sr=sr, tempo=tempo, beat_times=beat_times
    )
    return mtc.correct(notes, **kwargs)


# =====================================================
# 直接実行 (簡易動作確認)
# =====================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # サンプルノート (C メジャー風) を作って動作確認
    sample_notes = [
        (0.00, 0.45, 60, 100),   # C4 (コードトーン)
        (0.00, 0.45, 64, 90),    # E4 (コードトーン)
        (0.00, 0.45, 67, 85),    # G4 (コードトーン)
        (0.48, 0.02, 61, 40),    # C#4 短音・コード外 → 削除候補
        (0.50, 0.45, 62, 90),    # D4
        (1.00, 0.45, 65, 88),    # F4
        (1.52, 0.50, 55, 110),   # G3 (ベース)
        (1.52, 0.50, 56, 105),   # G#3 同時ベース → 削除候補
    ]
    chords = [
        (0.0, 1.0, 0, 'maj', 0.8),   # C maj
        (1.0, 1.0, 5, 'maj', 0.75),  # F maj
    ]

    print("--- 補正前 ---")
    for n in sample_notes:
        print(" ", n)

    refined = correct_by_chord(sample_notes, chords)
    refined = stabilize_bass(refined)
    print("--- 補正後 ---")
    for n in refined:
        print(" ", n)

    scored = score_confidence(refined, chords=chords,
                              key_root=0, key_mode='major')
    print("--- 信頼度 ---")
    for s in scored:  # noqa: E501 (keep original demo intact)
        print(f"  midi={s[2]:3d} conf={s[4]:.2f}")

# =====================================================
# ① 時系列安定化: ChordHistory
# =====================================================

class ChordHistory:
    """コード推定を時間方向で安定化する履歴バッファ。

    過去 window フレームのコード推定を保持し、
    多数決 (最頻値) で安定したコードを返す。
    信頼度が低い場合は前フレームのコードを維持。

    ① 時系列安定化: chord_history で多数決
    ② コード補正強化: confidence 導入 + 遷移平滑化
    """

    def __init__(self, window=8, min_confidence=0.40,
                 transition_penalty=0.15):
        """
        window             : 多数決対象の履歴フレーム数
        min_confidence     : これ未満は前フレームを維持
        transition_penalty : 直前コードと異なる場合のスコア罰則
        """
        self.window = window
        self.min_confidence = min_confidence
        self.transition_penalty = transition_penalty
        self._history = []          # [(root, quality, score), ...]
        self._stable = None         # 現在の安定コード

    def update(self, root, quality, score):
        """新しいフレームのコード推定を投入し、安定コードを返す。

        Returns:
            (root, quality, confidence) — 安定化後のコード
        """
        # 信頼度が低すぎる場合は前フレームを維持
        if score < self.min_confidence and self._stable is not None:
            self._history.append(self._stable)
        else:
            self._history.append((root, quality, score))

        if len(self._history) > self.window:
            self._history.pop(0)

        # 多数決 (root, quality) ペアの最頻値
        votes = Counter((r, q) for r, q, _ in self._history)
        (best_root, best_q), count = votes.most_common(1)[0]

        # スコア平均
        scores = [s for r, q, s in self._history
                  if r == best_root and q == best_q]
        avg_score = float(np.mean(scores)) if scores else 0.0

        # 遷移罰則: 直前と異なるコードは confidence を下げる
        if (self._stable is not None and
                (best_root, best_q) != self._stable[:2]):
            avg_score -= self.transition_penalty

        self._stable = (best_root, best_q, max(0.0, avg_score))
        return self._stable

    def reset(self):
        self._history.clear()
        self._stable = None


def estimate_chords_stable(audio, sr=44100, beat_times=None,
                            hop_length=512, window=8,
                            min_confidence=0.40):
    """① + ② 時系列安定化付きコード推定。

    estimate_chords() を ChordHistory で安定化する。

    Returns:
        [(onset_s, dur_s, root_pc, quality, confidence), ...]
    """
    raw = estimate_chords(audio, sr=sr, beat_times=beat_times,
                          hop_length=hop_length)
    if not raw:
        return raw

    history = ChordHistory(window=window, min_confidence=min_confidence)
    stable_events = []
    for (onset, dur, root, q, score) in raw:
        s_root, s_q, s_conf = history.update(root, q, score)
        stable_events.append((onset, dur, s_root, s_q, s_conf))

    # 連続する同コードを再マージ
    return _merge_adjacent_chords(stable_events)


# =====================================================
# ③ セクション別キー推定: SectionKeyEstimator
# =====================================================

class SectionKeyEstimator:
    """曲をセクション (複数小節単位) に分割し、
    セクションごとにキーを再推定する。

    転調を検出し、セクション単位でスケールフィルタを適用可能。
    """

    def __init__(self, section_sec=16.0, min_confidence=0.45):
        """
        section_sec    : 1 セクションの長さ (秒)
        min_confidence : これ未満のセクションは前セクションのキーを引き継ぐ
        """
        self.section_sec = section_sec
        self.min_confidence = min_confidence
        self.sections = []   # [(start, end, key_root, key_mode, score), ...]

    def analyze(self, audio, sr=44100):
        """音声全体を解析してセクション別キーを推定する。

        Returns:
            self (メソッドチェーン可)
        """
        total = len(audio) / sr
        self.sections = []
        prev = (0, 'major')

        hop = int(self.section_sec * sr)
        for i, start in enumerate(range(0, len(audio), hop)):
            seg = audio[start:start + hop]
            if len(seg) < sr:          # 1秒未満はスキップ
                break
            root, mode, score = estimate_key(seg, sr=sr)
            end = min((i + 1) * self.section_sec, total)
            if score < self.min_confidence:
                root, mode = prev
                score = 0.0
            self.sections.append((i * self.section_sec, end,
                                  root, mode, score))
            prev = (root, mode)

        log.info("SectionKeyEstimator: %d sections", len(self.sections))
        return self

    def key_at(self, t):
        """時刻 t のキーを返す。"""
        for (start, end, root, mode, score) in self.sections:
            if start <= t < end:
                return (root, mode, score)
        if self.sections:
            return self.sections[-1][2:]  # type: ignore
        return (0, 'major', 0.0)

    def filter_notes(self, notes, tolerance=0.15):
        """セクション別キーでスケールフィルタを適用。"""
        if not self.sections:
            return notes
        out = []
        for (t, d, m, v) in notes:
            root, mode, score = self.key_at(t)
            if score < self.min_confidence:
                out.append((t, d, m, v))  # 信頼低 → 通過
                continue
            allowed = _scale_pc_set(root, mode)
            pc = int(m) % 12
            if pc in allowed:
                out.append((t, d, m, v))
            elif d <= tolerance:
                out.append((t, d, m, max(1, int(v * 0.7))))
        return out


# =====================================================
# ④ 動的信頼度フィルタ
# =====================================================

def filter_by_dynamic_threshold(notes, chords=None,
                                 key_root=None, key_mode=None,
                                 top_k_per_window=5,
                                 window_sec=0.1):
    """④ ノート信頼度を動的閾値でフィルタする。

    固定 threshold ではなく、各時間窓内で上位 top_k のみ残す。
    + 平均 confidence 以下のノートも削除。

    Args:
        notes           : [(onset, dur, midi, vel), ...]
        top_k_per_window: 時間窓内で残す最大ノート数
        window_sec      : 時間窓の幅 (秒)
    """
    if not notes:
        return notes

    scored = score_confidence(notes, chords=chords,
                              key_root=key_root, key_mode=key_mode)
    if not scored:
        return notes

    # 平均 confidence
    avg_conf = float(np.mean([s[4] for s in scored]))
    min_thr = max(0.15, avg_conf * 0.7)  # 平均の70%以下は削除

    notes_sorted = sorted(scored, key=lambda n: n[0])
    result = []
    n = len(notes_sorted)
    i = 0
    while i < n:
        t_i = notes_sorted[i][0]
        group = []
        j = i
        while j < n and notes_sorted[j][0] - t_i < window_sec:
            group.append(notes_sorted[j])
            j += 1
        # top-K and above-threshold
        group_pass = [rec for rec in group if rec[4] >= min_thr]
        group_pass.sort(key=lambda r: r[4], reverse=True)
        for rec in group_pass[:top_k_per_window]:
            result.append((rec[0], rec[1], rec[2], rec[3]))
        i = j

    return sorted(result, key=lambda n: n[0])


# =====================================================
# ⑤ 和音構造の強制 (Triad Enforcer)
# =====================================================

def enforce_triad_structure(notes, chords,
                             max_voices=4,
                             keep_bass=True,
                             bass_range=(0, 55)):
    """⑤ 同時発音をトライアドベースに整理する。

    同時刻 (50ms窓) に max_voices を超える音がある場合:
    1. コードトーン (root/3rd/5th) を優先して残す
    2. 余分なテンション音 (7th以上) を削る
    3. bass_range 内のノートは keep_bass=True なら保護

    Args:
        max_voices : 同時最大声部数 (4 = triad + bass)
    """
    if not notes or not chords:
        return notes

    notes_sorted = sorted(notes, key=lambda n: n[0])
    result = []
    n = len(notes_sorted)
    window = 0.05
    i = 0

    while i < n:
        t_i = notes_sorted[i][0]
        group = []
        j = i
        while j < n and notes_sorted[j][0] - t_i < window:
            group.append(notes_sorted[j])
            j += 1

        if len(group) <= max_voices:
            result.extend(group)
        else:
            active = _active_chord_at(chords, t_i)
            if active is None:
                result.extend(group[:max_voices])
                i = j
                continue

            root, q, _ = active
            triad_ivs = CHORD_INTERVALS.get(q, [0, 4, 7])[:3]
            triad_pcs = {(root + iv) % 12 for iv in triad_ivs}
            chord_pcs = _chord_pitch_classes(root, q)

            def _priority(note):
                t, d, m, v = note
                pc = int(m) % 12
                in_bass = keep_bass and bass_range[0] <= m <= bass_range[1]
                if in_bass:
                    return (0, -v)       # bass 最優先
                if pc in triad_pcs:
                    return (1, -v)       # triad
                if pc in chord_pcs:
                    return (2, -v)       # 7th など拡張コード
                return (3, -v)           # コード外

            group.sort(key=_priority)
            result.extend(group[:max_voices])

        i = j

    return sorted(result, key=lambda n: n[0])


# =====================================================
# ⑥ ベースライン最適化
# =====================================================

def optimize_bass(notes, chords, bass_range=(24, 55),
                  max_jump=12, root_snap_strength=0.8):
    """⑥ ベースノートをコードルートに寄せ、急激なジャンプを抑制する。

    処理:
    1. ベースノートをコードルートの最近傍オクターブにスナップ
    2. 隣接ベースノート間のジャンプが max_jump 以上 → 中間を補間
    3. 連続する同ピッチをマージ (滑らか化)

    Args:
        root_snap_strength : 0-1 (1.0=完全にルートへ移動)
    """
    if not notes or not chords:
        return notes

    bass = [(t, d, m, v) for (t, d, m, v) in notes
            if bass_range[0] <= m <= bass_range[1]]
    other = [(t, d, m, v) for (t, d, m, v) in notes
             if not (bass_range[0] <= m <= bass_range[1])]

    if not bass:
        return notes

    bass.sort(key=lambda n: n[0])

    # 1. コードルートへのスナップ
    snapped = []
    for (t, d, m, v) in bass:
        active = _active_chord_at(chords, t)
        if active is None:
            snapped.append((t, d, m, v))
            continue
        root_pc, q, score = active
        if score < 0.35:
            snapped.append((t, d, m, v))
            continue

        # ルートの最近傍オクターブを探す
        best_m = m
        best_dist = 999
        for oct_ in range(1, 7):
            candidate = root_pc + 12 * oct_
            if bass_range[0] <= candidate <= bass_range[1]:
                dist = abs(candidate - m)
                if dist < best_dist:
                    best_dist = dist
                    best_m = candidate

        if best_dist <= 6:  # 半音6以内なら部分スナップ
            new_m = int(round(m + (best_m - m) * root_snap_strength))
            new_m = max(bass_range[0], min(bass_range[1], new_m))
        else:
            new_m = m
        snapped.append((t, d, new_m, v))

    # 2. 急激なジャンプ抑制
    smooth = [snapped[0]]
    for i in range(1, len(snapped)):
        prev = smooth[-1]
        cur = snapped[i]
        jump = abs(cur[2] - prev[2])
        if jump > max_jump:
            # ジャンプが大きすぎるなら 1 オクターブ分戻す
            direction = 1 if cur[2] > prev[2] else -1
            new_m = cur[2] - direction * 12
            new_m = max(bass_range[0], min(bass_range[1], new_m))
            smooth.append((cur[0], cur[1], new_m, cur[3]))
        else:
            smooth.append(cur)

    return sorted(smooth + other, key=lambda n: n[0])


# =====================================================
# ⑦ リズム強化 (最小ノート長 + 強制量子化)
# =====================================================

def stabilize_rhythm(notes, tempo=120.0, beat_times=None,
                     min_dur_sec=0.06, subdivision=16,
                     strength=0.8):
    """⑦ リズム安定化の強化版。

    1. min_dur_sec 未満のノートを削除
    2. 強めの量子化 (strength=0.8)
    3. 同ピッチ短間隔ノートをマージ
    """
    if not notes:
        return notes

    # 1. 最小長フィルタ
    filtered = [(t, d, m, v) for (t, d, m, v) in notes
                if d >= min_dur_sec]

    # 2. 量子化
    filtered = quantize_rhythm(filtered, tempo=tempo,
                               beat_times=beat_times,
                               subdivision=subdivision,
                               strength=strength)

    # 3. 同ピッチ短間隔マージ (gap < 1/4拍)
    beat_dur = 60.0 / max(tempo, 30)
    gap_thr = beat_dur / 4
    by_pitch = {}
    for n in filtered:
        by_pitch.setdefault(n[2], []).append(n)

    out = []
    for midi, group in by_pitch.items():
        group.sort(key=lambda n: n[0])
        merged = [list(group[0])]
        for note in group[1:]:
            last = merged[-1]
            gap = note[0] - (last[0] + last[1])
            if 0 < gap < gap_thr:
                last[1] = note[0] + note[1] - last[0]
                last[3] = max(last[3], note[3])
            else:
                merged.append(list(note))
        out.extend(tuple(m) for m in merged)

    return sorted(out, key=lambda n: n[0])


# =====================================================
# ⑧ モデル出力スムージング (予測確率行列に適用)
# =====================================================

def smooth_predictions(pitch_prob, onset_prob,
                        method='ema', alpha=0.4, window=3):
    """⑧ フレーム確率行列に時間方向スムージングを適用する。

    mimikopi.py の _decode_predictions に渡す前に使用。

    Args:
        pitch_prob : (T, P) ndarray 0-1
        onset_prob : (T, P) ndarray 0-1
        method     : 'ema' (指数移動平均) or 'moving_avg'
        alpha      : EMA のスムージング係数 (小さいほど強く平滑化)
        window     : moving_avg のウィンドウ幅

    Returns:
        (smoothed_pitch_prob, smoothed_onset_prob) どちらも (T, P)
    """
    if pitch_prob is None or pitch_prob.size == 0:
        return pitch_prob, onset_prob

    def _smooth(arr):
        T, P = arr.shape
        if method == 'ema':
            out = arr.copy()
            for t in range(1, T):
                out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
            return out
        else:  # moving_avg
            kernel = np.ones(window) / window
            out = np.apply_along_axis(
                lambda col: np.convolve(col, kernel, mode='same'),
                axis=0, arr=arr
            )
            return np.clip(out, 0.0, 1.0)

    p_smooth = _smooth(pitch_prob)
    # onset はあまり平滑化しすぎると検出が鈍くなるので控えめに
    o_alpha = min(alpha * 1.5, 0.8)
    o_smooth = _smooth(onset_prob) if method != 'ema' else (
        onset_prob * o_alpha + np.roll(onset_prob, 1, axis=0) * (1 - o_alpha)
    )
    return p_smooth.astype(np.float32), o_smooth.astype(np.float32)


# =====================================================
# ⑨ エラー耐性: 無音 / ノイズ検出
# =====================================================

def filter_silence_noise(notes, audio, sr=44100,
                          silence_rms=0.005,
                          noise_dur_max=0.04,
                          hop_length=512):
    """⑨ 無音区間の誤検出ノートと突発ノイズを除去する。

    処理:
    1. 音声の RMS を時間軸に沿って計算
    2. RMS < silence_rms の区間で発生したノートを削除
    3. dur < noise_dur_max かつ その前後が無音の突発ノートを削除

    Args:
        audio         : 元音声 1D ndarray
        silence_rms   : 無音判定閾値
        noise_dur_max : ノイズノート判定の最大デュレーション (秒)
    """
    if not notes or audio is None or len(audio) == 0:
        return notes

    try:
        import librosa
        rms = librosa.feature.rms(y=audio, frame_length=2048,
                                   hop_length=hop_length)[0]
    except Exception:
        return notes

    frame_sec = hop_length / sr

    def _rms_at(t):
        f = int(t / frame_sec)
        f = max(0, min(len(rms) - 1, f))
        return float(rms[f])

    out = []
    notes_sorted = sorted(notes, key=lambda n: n[0])
    n = len(notes_sorted)

    for i, (t, d, m, v) in enumerate(notes_sorted):
        # 無音区間チェック (ノート中間時刻)
        mid_t = t + d / 2
        if _rms_at(mid_t) < silence_rms:
            continue

        # 突発ノイズチェック
        if d <= noise_dur_max:
            prev_rms = _rms_at(t - 0.05) if t > 0.05 else 0.0
            next_rms = _rms_at(t + d + 0.05)
            if prev_rms < silence_rms * 3 and next_rms < silence_rms * 3:
                continue

        out.append((t, d, m, v))

    log.debug("filter_silence_noise: %d → %d", len(notes), len(out))
    return out


# =====================================================
# MusicTheoryCorrector の拡張版 (9機能統合)
# =====================================================

class AdvancedMusicTheoryCorrector(MusicTheoryCorrector):
    """MusicTheoryCorrector に ①〜⑨ の拡張を加えたクラス。

    継承で既存機能を維持しつつ新機能を追加。
    """

    def __init__(self, audio=None, sr=44100, tempo=120.0,
                 beat_times=None, hop_length=512,
                 chord_window=8, section_sec=16.0):
        """
        chord_window : ChordHistory のウィンドウ (フレーム数)
        section_sec  : SectionKeyEstimator のセクション長 (秒)
        """
        # ① ② 安定化コード推定
        self._chord_history = ChordHistory(window=chord_window)
        # ③ セクション別キー推定
        self._section_key = SectionKeyEstimator(section_sec=section_sec)

        # 親の __init__ は chords / key_root / key_mode を設定する
        # ただし chords は安定化版で上書きする
        super().__init__(
            audio=audio, sr=sr, tempo=tempo,
            beat_times=beat_times, hop_length=hop_length,
        )

        if audio is not None and len(audio) > 0:
            # 安定化コード推定で上書き
            self.chords = estimate_chords_stable(
                audio, sr=sr, beat_times=beat_times,
                hop_length=hop_length,
                window=chord_window,
            )
            # セクション別キー推定
            self._section_key.analyze(audio, sr=sr)

    def correct(self, notes, audio=None,
                apply_scale=True,
                apply_chord=True,
                apply_bass=True,
                apply_rhythm=True,
                apply_confidence=True,
                apply_harmony=True,
                apply_triad=True,
                apply_bass_opt=True,
                apply_silence=True,
                confidence_threshold=0.28,
                max_polyphony=4,
                quantize_strength=0.65,
                top_k_per_window=5):
        """拡張フルパイプライン。

        親クラスの correct() を置き換え、9機能を全適用。
        """
        if not notes:
            return notes

        log.info("AdvancedMusicTheoryCorrector.correct: input=%d", len(notes))

        # ⑨ 無音・ノイズ除去 (最初に適用して後続の負荷を減らす)
        raw_audio = (audio if audio is not None else getattr(self, "audio", None))
        if apply_silence and raw_audio is not None:
            notes = filter_silence_noise(notes, raw_audio, sr=self.sr)

        # ④ ベース安定化
        if apply_bass:
            notes = stabilize_bass(notes)

        # ③ セクション別スケールフィルタ
        if apply_scale:
            if self._section_key.sections:
                notes = self._section_key.filter_notes(notes)
            elif self.key_score > 0.3:
                notes = filter_by_scale(notes, self.key_root, self.key_mode)

        # ② コード補正 (安定化コード使用)
        if apply_chord and self.chords:
            notes = correct_by_chord(
                notes, self.chords, max_polyphony=max_polyphony
            )

        # ⑤ トライアド構造強制
        if apply_triad and self.chords:
            notes = enforce_triad_structure(
                notes, self.chords, max_voices=max_polyphony
            )

        # ⑦ 和音優先
        if apply_harmony and self.chords:
            notes = prioritize_harmony(notes, self.chords)

        # ⑥ ベースライン最適化
        if apply_bass_opt and self.chords:
            notes = optimize_bass(notes, self.chords)

        # ⑦ リズム安定化 (強化版)
        if apply_rhythm:
            notes = stabilize_rhythm(
                notes, tempo=self.tempo,
                beat_times=self.beat_times,
                strength=quantize_strength,
            )

        # ④ 動的信頼度フィルタ
        if apply_confidence:
            key_root = (self.key_root if self.key_score > 0.3 else None)
            key_mode = (self.key_mode if self.key_score > 0.3 else None)
            notes = filter_by_dynamic_threshold(
                notes,
                chords=self.chords,
                key_root=key_root,
                key_mode=key_mode,
                top_k_per_window=top_k_per_window,
            )

        log.info("AdvancedMusicTheoryCorrector.correct: output=%d", len(notes))
        return sorted(notes, key=lambda n: n[0])


def correct_notes_advanced(notes, audio=None, sr=44100, tempo=120.0,
                            beat_times=None, **kwargs):
    """⑨機能統合ワンショット関数。"""
    mtc = AdvancedMusicTheoryCorrector(
        audio=audio, sr=sr, tempo=tempo, beat_times=beat_times,
    )
    return mtc.correct(notes, audio=audio, **kwargs)


# =====================================================
# 最終チューニング層: ヒューマナイゼーション
# =====================================================
# 目的: 「正しい音」ではなく「人間が演奏したように聴こえる」出力
# 構造変更なし、既存パイプラインの末尾に挿入する後処理層

# ① 遅延許容型安定化 (look-ahead 多数決)
class DelayedStabilizer:
    """2〜3 フレームの遅延を許容して、未来も含めた多数決で確定する。

    リアルタイムでない後処理なら全体を見渡せるので、
    各時刻 t の決定を [t-W, t+W] の窓で多数決する。
    """

    def __init__(self, look_ahead=3, look_back=3):
        self.look_ahead = look_ahead
        self.look_back = look_back

    def stabilize_chords(self, chord_events):
        """chord_events: [(onset, dur, root, q, score), ...] を平滑化"""
        if not chord_events:
            return chord_events
        n = len(chord_events)
        out = []
        for i in range(n):
            lo = max(0, i - self.look_back)
            hi = min(n, i + self.look_ahead + 1)
            window = chord_events[lo:hi]
            # スコア重み付き多数決
            votes = {}
            for (_, _, r, q, s) in window:
                key = (r, q)
                votes[key] = votes.get(key, 0.0) + max(s, 0.05)
            best = max(votes.items(), key=lambda x: x[1])
            (root, qual), weight = best
            onset, dur, _, _, score = chord_events[i]
            confidence = weight / sum(votes.values())
            out.append((onset, dur, root, qual, confidence))
        return _merge_adjacent_chords(out)


# ② ノート持続補正
def enforce_min_duration(notes, min_dur=0.08, extend_to=0.10):
    """最低ノート長を保証。短いノートを extend_to まで延長。

    Args:
        min_dur   : これ未満のノートを処理対象にする
        extend_to : 延長後の長さ (秒)
    """
    if not notes:
        return notes
    out = []
    for (t, d, m, v) in notes:
        if d < min_dur:
            out.append((t, max(d, extend_to), m, v))
        else:
            out.append((t, d, m, v))
    return out


# ③ コード遷移制御 (前フレーム優先・微差は維持)
def smooth_chord_transitions(chord_events, hysteresis=0.10):
    """confidence 差が hysteresis 未満なら前のコードを維持する。

    急激な遷移を抑制し、自然なコード持続を作る。
    """
    if not chord_events:
        return chord_events
    out = [chord_events[0]]
    prev_root, prev_q, prev_conf = chord_events[0][2:5]
    for ev in chord_events[1:]:
        onset, dur, root, q, conf = ev
        if (root, q) != (prev_root, prev_q):
            # 信頼度差が小さいなら前を維持
            if conf - prev_conf < hysteresis:
                out.append((onset, dur, prev_root, prev_q, prev_conf))
                continue
        out.append(ev)
        prev_root, prev_q, prev_conf = root, q, conf
    return _merge_adjacent_chords(out)


# ④ ベースライン滑らか化 (±5半音以内に制限)
def smooth_bass_strict(notes, bass_range=(24, 55), max_jump=5):
    """ベースの隣接ノート間ジャンプを max_jump 半音以内に制限。

    超える場合は前ノートに半音単位で寄せる (8度以上は1オクターブ補正)。
    """
    if not notes:
        return notes
    bass = sorted([n for n in notes if bass_range[0] <= n[2] <= bass_range[1]],
                  key=lambda x: x[0])
    other = [n for n in notes if not (bass_range[0] <= n[2] <= bass_range[1])]
    if len(bass) < 2:
        return notes

    smooth = [bass[0]]
    for cur in bass[1:]:
        prev = smooth[-1]
        diff = cur[2] - prev[2]
        if abs(diff) > max_jump:
            # オクターブ単位で補正
            shift = -12 if diff > 0 else 12
            new_m = cur[2] + shift
            # それでも超える場合は max_jump にクランプ
            if abs(new_m - prev[2]) > max_jump:
                new_m = prev[2] + (max_jump if diff > 0 else -max_jump)
            new_m = max(bass_range[0], min(bass_range[1], new_m))
            smooth.append((cur[0], cur[1], new_m, cur[3]))
        else:
            smooth.append(cur)
    return sorted(smooth + other, key=lambda n: n[0])


# ⑤ 動的閾値 (mean + α * std)
def adaptive_threshold(values, alpha=-0.5, lower=0.10, upper=0.85):
    """値ベクトルから mean + α * std で閾値を算出する。

    α < 0 → 平均より下を許容 (緩い)
    α > 0 → 平均より上のみ採用 (厳しい)
    """
    if values is None or len(values) == 0:
        return lower
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    thr = mean + alpha * std
    return float(np.clip(thr, lower, upper))


def filter_by_adaptive_confidence(scored_notes, alpha=-0.3):
    """⑤ score_confidence の出力を mean + α*std で動的に閾値処理する。"""
    if not scored_notes:
        return []
    confs = [s[4] for s in scored_notes]
    thr = adaptive_threshold(confs, alpha=alpha, lower=0.18, upper=0.55)
    out = []
    for rec in scored_notes:
        if len(rec) == 5 and rec[4] >= thr:
            out.append(rec[:4])
        elif len(rec) == 4:
            out.append(rec)
    log.debug("filter_by_adaptive_confidence: thr=%.3f, kept %d/%d",
              thr, len(out), len(scored_notes))
    return out


# ⑥ リズムの人間化 (微小ジッター)
def humanize_rhythm(notes, jitter_sec=0.010, vel_jitter=4, seed=None):
    """完全量子化を緩めるため、各ノートに微小ランダムを加える。

    Args:
        jitter_sec : onset の最大ずらし量 (±jitter_sec)
        vel_jitter : velocity の最大ずらし量 (±vel_jitter)
        seed       : 乱数シード (再現性が必要な場合)
    """
    if not notes:
        return notes
    rng = np.random.default_rng(seed)
    out = []
    for (t, d, m, v) in notes:
        dt = float(rng.uniform(-jitter_sec, jitter_sec))
        dv = int(rng.integers(-vel_jitter, vel_jitter + 1))
        new_t = max(0.0, t + dt)
        new_v = int(np.clip(v + dv, 1, 127))
        out.append((new_t, d, m, new_v))
    return sorted(out, key=lambda n: n[0])


# ⑦ MIDI 最終整形 (同時発音制限・音域・velocity 強弱)
def finalize_midi_shape(notes, max_polyphony=5,
                        midi_range=(21, 108),
                        accent_beats=None, beat_dur=0.5,
                        accent_boost=12):
    """同時発音制限・音域フィルタ・拍頭にアクセント。

    Args:
        max_polyphony : 同時最大ノート数 (50ms 窓)
        midi_range    : 許容 MIDI 範囲 (A0=21, C8=108)
        accent_beats  : アクセントを付ける拍時刻 (秒). None なら beat_dur 等分
        beat_dur      : ビート長 (accent_beats が None のとき使用)
        accent_boost  : 拍頭ノートの velocity 加算量
    """
    if not notes:
        return notes

    # 音域フィルタ
    notes = [(t, d, m, v) for (t, d, m, v) in notes
             if midi_range[0] <= m <= midi_range[1]]

    # 同時発音制限 (vel 上位を残す)
    notes.sort(key=lambda n: n[0])
    result = []
    n = len(notes)
    window = 0.05
    i = 0
    while i < n:
        t_i = notes[i][0]
        group = [notes[i]]
        j = i + 1
        while j < n and notes[j][0] - t_i < window:
            group.append(notes[j])
            j += 1
        if len(group) <= max_polyphony:
            result.extend(group)
        else:
            group.sort(key=lambda x: -x[3])  # velocity 降順
            result.extend(group[:max_polyphony])
        i = j

    # 拍頭アクセント
    if accent_beats is None:
        if not result:
            return result
        total = max(n[0] + n[1] for n in result)
        accent_beats = np.arange(0, total + beat_dur, beat_dur)
    accent_beats = np.asarray(accent_beats, dtype=float)

    accented = []
    for (t, d, m, v) in result:
        # 最近傍ビートとの距離
        dist = float(np.min(np.abs(accent_beats - t)))
        if dist < 0.04:    # 拍頭(40ms以内)
            new_v = int(np.clip(v + accent_boost, 30, 127))
        elif dist < 0.10:  # 拍裏付近 → 弱める
            new_v = int(np.clip(v - 3, 25, 127))
        else:
            new_v = v
        accented.append((t, d, m, new_v))
    return sorted(accented, key=lambda n: n[0])


# ⑧ 全体最適化 (ヒューマナイゼーション統合)
def humanize_notes(notes, audio=None, sr=44100, tempo=120.0,
                   beat_times=None, chords=None,
                   min_dur=0.08, jitter_sec=0.010, vel_jitter=4,
                   max_polyphony=5, max_bass_jump=5,
                   midi_range=(21, 108), accent_boost=10,
                   apply_jitter=True, apply_accent=True,
                   apply_bass_smooth=True, apply_min_dur=True,
                   apply_polyphony=True, seed=None):
    """ヒューマナイゼーションをまとめて適用する後処理関数。

    既存パイプラインの末尾に挿入することで、
    出力を「人間が演奏した」感じに整える。

    処理順:
      ② 最低ノート長の保証
      ④ ベースライン滑らか化 (±5半音)
      ⑦ 同時発音制限・音域フィルタ・拍頭アクセント
      ⑥ リズム微小ジッター + velocity ジッター
    """
    if not notes:
        return notes

    if apply_min_dur:
        notes = enforce_min_duration(notes, min_dur=min_dur,
                                      extend_to=min_dur + 0.02)

    if apply_bass_smooth:
        notes = smooth_bass_strict(notes, max_jump=max_bass_jump)

    if apply_polyphony:
        notes = finalize_midi_shape(
            notes, max_polyphony=max_polyphony,
            midi_range=midi_range,
            accent_beats=beat_times,
            beat_dur=60.0 / max(tempo, 30.0),
            accent_boost=accent_boost if apply_accent else 0,
        )

    if apply_jitter:
        notes = humanize_rhythm(notes, jitter_sec=jitter_sec,
                                vel_jitter=vel_jitter, seed=seed)

    log.info("humanize_notes: output=%d notes", len(notes))
    return notes


def correct_and_humanize(notes, audio=None, sr=44100, tempo=120.0,
                         beat_times=None, seed=None,
                         humanize_kwargs=None, correct_kwargs=None):
    """⑨機能補正 + ヒューマナイズ をワンショットで適用する。"""
    correct_kwargs = correct_kwargs or {}
    humanize_kwargs = humanize_kwargs or {}

    mtc = AdvancedMusicTheoryCorrector(
        audio=audio, sr=sr, tempo=tempo, beat_times=beat_times,
    )

    # 安定化済みコードを使ってヒューマナイズも一貫性を保つ
    if mtc.chords:
        # ① 遅延許容型安定化 + ③ 遷移平滑化
        stabilizer = DelayedStabilizer(look_ahead=3, look_back=3)
        stable = stabilizer.stabilize_chords(mtc.chords)
        mtc.chords = smooth_chord_transitions(stable, hysteresis=0.10)

    refined = mtc.correct(notes, audio=audio, **correct_kwargs)
    final = humanize_notes(refined, audio=audio, sr=sr,
                           tempo=tempo, beat_times=beat_times,
                           chords=mtc.chords, seed=seed,
                           **humanize_kwargs)
    return final
