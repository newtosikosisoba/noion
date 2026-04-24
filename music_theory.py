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
    for s in scored:
        print(f"  midi={s[2]:3d} conf={s[4]:.2f}")
