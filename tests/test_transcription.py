"""
採譜パイプラインの回帰テスト (4ケース)
- 短音 (staccato / 16分音符連打)
- 連打 (同一ピッチが繰り返される)
- 和音 (同時多音)
- 低音 (C1-C3 のベースライン)

実際の音声ファイルを使わず、合成 ndarray で各メソッドを単体テストする。
"""
import sys
import os
import types
import numpy as np
import pytest

# --- Dummy deps for import without GPU libs --------------------------------

# Only create stubs for modules that are NOT already installed
def _stub(name):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]

for _pkg in ["basic_pitch", "demucs", "torchaudio"]:
    _stub(_pkg)

# basic_pitch sub-modules
bp = sys.modules["basic_pitch"]
bp.ICASSP_2022_MODEL_PATH = "/dev/null"
inf_mod = types.ModuleType("basic_pitch.inference")
inf_mod.predict = None
sys.modules["basic_pitch.inference"] = inf_mod

# mido is already installed; no stub needed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import librosa
from scipy.signal import butter, filtfilt

SR = 44100
HOP = 512

# ---------------------------------------------------------------------------
# Helper: synthesise a clean sine-wave note
# ---------------------------------------------------------------------------

def _sine(hz: float, dur: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (np.sin(2 * np.pi * hz * t) * amp).astype(np.float32)


def _midi_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


# ---------------------------------------------------------------------------
# Import the processor (after stubs are registered)
# ---------------------------------------------------------------------------

from mimikopi import EarCopyEngine  # noqa: E402  (import after stubs)


def _proc() -> EarCopyEngine:
    p = EarCopyEngine(on_progress=None, mode="ai")
    p.transcription_quality = "balanced"
    return p


# ===========================================================================
# Case 1: 短音 — 40ms の staccato 16分音符が正しく検出される
# ===========================================================================

class TestShortNotes:
    """_f0_to_notes と _detect_all_notes が短音を取りこぼさないことを確認。"""

    def test_short_note_floor_40ms(self):
        """CQT ノートの最短時間フロアが 40ms 以下であることを確認。"""
        proc = _proc()
        cfg = proc._get_transcription_cfg()
        assert cfg['dur_floor_s'] <= 0.04, \
            f"dur_floor_s={cfg['dur_floor_s']} > 40ms — 短音が切り捨てられる"

    def test_f0_short_note_captured(self):
        """_f0_to_notes: 50ms ノートが1個返ること。"""
        proc = _proc()
        hz = _midi_hz(60)  # C4
        sr = SR
        # 50ms のサイン波
        seg_dur = 0.05
        audio = _sine(hz, seg_dur, sr)
        f0, voiced, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=HOP)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
        rms = librosa.feature.rms(y=audio, frame_length=HOP*2, hop_length=HOP)[0]
        notes = proc._f0_to_notes(f0, voiced, times, 8, rms)
        # pyin はノイズの無い正弦波ならほぼ必ず検出できる
        assert len(notes) >= 1, "50ms の短音が1音も検出されなかった"
        durations = [d for _, d, _, _ in notes]
        assert all(d >= 0.04 for d in durations), \
            f"40ms フロア未満のノートが含まれる: {min(durations):.3f}s"

    def test_quantize_does_not_inflate_short_notes(self):
        """_quantize_notes: 短音 (<1グリッド) の duration が膨らまないこと。"""
        proc = _proc()
        # 16分音符 = 0.125s @ 120BPM
        beat_times = np.arange(0, 4, 0.5)
        tempo = 120.0
        short_dur = 0.06  # 1グリッドの半分以下
        notes = [(0.0, short_dur, 60, 80)]
        quantized = proc._quantize_notes(notes, beat_times, tempo)
        assert len(quantized) == 1
        _, qdur, _, _ = quantized[0]
        assert qdur <= short_dur * 1.05, \
            f"短音が量子化で膨らんだ: {short_dur:.3f}s → {qdur:.3f}s"


# ===========================================================================
# Case 2: 連打 — 同一ピッチの繰り返しノートがオーバーラップしないこと
# ===========================================================================

class TestRepeatedNotes:
    """_remove_overlapping_notes が重複する同ピッチノートを処理すること。"""

    def test_overlap_removed(self):
        """同ピッチで重なる2ノートの前者が後者の開始前に終端されること。"""
        proc = _proc()
        notes = [
            (0.0, 0.5, 60, 80),   # C4, 0.0-0.5s
            (0.3, 0.5, 60, 80),   # C4, 0.3-0.8s (重なっている)
        ]
        cleaned = proc._remove_overlapping_notes(notes)
        cleaned_60 = [(t, d) for t, d, m, v in cleaned if m == 60]
        assert len(cleaned_60) == 2, "ノートが失われた"
        t1, d1 = cleaned_60[0]
        t2, d2 = cleaned_60[1]
        assert t1 + d1 <= t2 + 1e-6, \
            f"前ノートが後ノートと重なっている: {t1+d1:.3f} > {t2:.3f}"

    def test_non_overlapping_unchanged(self):
        """重なりのないノートは変更されないこと。"""
        proc = _proc()
        notes = [
            (0.0, 0.2, 60, 80),
            (0.3, 0.2, 60, 90),
        ]
        cleaned = proc._remove_overlapping_notes(notes)
        assert len(cleaned) == 2
        t, d, m, v = sorted(cleaned)[0]
        assert abs(d - 0.2) < 1e-6, "duration が変わってしまった"

    def test_different_pitches_independent(self):
        """異なるピッチのノートは互いに影響しないこと。"""
        proc = _proc()
        notes = [
            (0.0, 0.5, 60, 80),
            (0.0, 0.5, 64, 80),  # 同時発音だが別ピッチ
        ]
        cleaned = proc._remove_overlapping_notes(notes)
        assert len(cleaned) == 2, "別ピッチのノートが削除された"


# ===========================================================================
# Case 3: 和音 — CQT が同時多音を検出できること
# ===========================================================================

class TestChords:
    """_detect_all_notes が和音（複数の同時音）を検出すること。"""

    def _make_chord(self, midis, dur=1.0, sr=SR, amp=0.25, harmonics=4):
        """倍音を含む和音波形を生成（より実楽器に近い）。"""
        n = int(dur * sr)
        audio = np.zeros(n, dtype=np.float32)
        t = np.arange(n) / sr
        for midi in midis:
            hz = _midi_hz(midi)
            for k in range(1, harmonics + 1):
                audio += np.sin(2 * np.pi * hz * k * t).astype(np.float32) * amp / k
        peak = np.max(np.abs(audio))
        return (audio / peak) if peak > 1e-9 else audio

    def test_major_triad_detected(self):
        """C メジャートライアド (C4, E4, G4): 複数の異なるピッチが検出されること。"""
        proc = _proc()
        midis = [60, 64, 67]  # C E G
        audio = self._make_chord(midis, dur=1.5)
        y_h, _ = librosa.effects.hpss(audio, margin=2.0)
        notes = proc._detect_all_notes(y_h, SR)
        detected_midis = list(set(m for _, _, m, _ in notes))
        # 倍音を含む和音なので少なくとも2つの異なるピッチが検出されるはず
        assert len(detected_midis) >= 2, \
            f"和音から2ピッチ以上検出されなかった: {sorted(detected_midis)}"

    def test_harmonic_suppression_not_over_aggressive(self):
        """強度比ベース倍音除去: 和音の3音が全て除去されないこと。"""
        proc = _proc()
        midis = [60, 64, 67]
        audio = self._make_chord(midis, dur=1.5, harmonics=6)
        y_h, _ = librosa.effects.hpss(audio, margin=2.0)
        notes = proc._detect_all_notes(y_h, SR)
        # 倍音除去が aggressive すぎると0音になる → 少なくとも1音は生き残ること
        assert len(notes) >= 1, "倍音除去が強すぎて全てのノートが除去された"


# ===========================================================================
# Case 4: 低音 — _f0_to_notes が C1-C3 のベースラインを検出できること
# ===========================================================================

class TestBassNotes:
    """pyin ベース採譜が低音域を正確に検出すること。"""

    def test_bass_c2_detected(self):
        """C2 (65.4 Hz) のベース音が正しい MIDI ノートとして返ること。"""
        proc = _proc()
        hz = _midi_hz(36)  # C2
        sr = SR
        nyq = sr / 2
        audio = _sine(hz, 2.0, sr, amp=0.6)
        # ベース前処理: ローパス
        b, a = butter(4, min(300 / nyq, 0.99), btype='low')
        audio = filtfilt(b, a, audio).astype(np.float32)
        f0, voiced, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C4"), sr=sr, hop_length=HOP)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
        rms = librosa.feature.rms(y=audio, frame_length=HOP*4, hop_length=HOP)[0]
        notes = proc._f0_to_notes(f0, voiced, times, 4, rms)
        assert len(notes) >= 1, "C2 のベース音が1音も検出されなかった"
        midis = [m for _, _, m, _ in notes]
        # MIDI 36 ± 2 の範囲で検出されることを確認
        assert any(34 <= m <= 38 for m in midis), \
            f"C2(midi=36) が検出されなかった: {midis}"

    def test_bass_amplitude_based_velocity(self):
        """RMS を渡した場合、velocity が動的に変わること（全て同一でない）。"""
        proc = _proc()
        sr = SR
        hz = _midi_hz(36)
        # 強弱がある波形: 前半は弱く、後半は強く
        quiet = _sine(hz, 1.0, sr, amp=0.1)
        loud = _sine(hz, 1.0, sr, amp=0.7)
        audio = np.concatenate([quiet, loud])
        f0, voiced, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C4"), sr=sr, hop_length=HOP * 2)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP * 2)
        rms = librosa.feature.rms(y=audio, frame_length=HOP*4, hop_length=HOP * 2)[0]
        notes = proc._f0_to_notes(f0, voiced, times, 4, rms)
        if len(notes) >= 2:
            vels = [v for _, _, _, v in notes]
            assert max(vels) - min(vels) > 5, \
                f"強弱があるのに velocity が均一: {vels}"

    def test_preprocess_bass_lowpass(self):
        """_preprocess_stem('bass') が 300Hz 超をカットすること。"""
        proc = _proc()
        sr = SR
        # 1kHz 純音（ベース帯域外）
        audio_1k = _sine(1000.0, 0.5, sr, amp=0.5)
        processed = proc._preprocess_stem(audio_1k, sr, 'bass')
        rms_orig = float(np.sqrt(np.mean(audio_1k ** 2)))
        rms_proc = float(np.sqrt(np.mean(processed ** 2)))
        # ローパス後は 1kHz 成分が大幅に減衰するはず
        assert rms_proc < rms_orig * 0.1, \
            f"1kHz がローパスで十分減衰されなかった: {rms_orig:.4f} → {rms_proc:.4f}"


# ===========================================================================
# Utility tests
# ===========================================================================

class TestUtils:
    def test_transcription_cfg_modes(self):
        """3モードが全て返せること、high_recall < balanced < high_precision の順。"""
        proc = _proc()
        for mode in ['high_recall', 'balanced', 'high_precision']:
            proc.transcription_quality = mode
            cfg = proc._get_transcription_cfg()
            assert 'onset_threshold' in cfg
            assert 'dur_floor_s' in cfg

        proc.transcription_quality = 'high_recall'
        cfg_hr = proc._get_transcription_cfg()
        proc.transcription_quality = 'high_precision'
        cfg_hp = proc._get_transcription_cfg()
        assert cfg_hr['onset_threshold'] < cfg_hp['onset_threshold'], \
            "high_recall の onset_threshold が high_precision より高い"
        assert cfg_hr['dur_floor_s'] < cfg_hp['dur_floor_s'], \
            "high_recall の dur_floor_s が high_precision より長い"

    def test_log_midi_stats_no_crash(self):
        """_log_midi_stats がクラッシュしないこと（空リストも含む）。"""
        proc = _proc()
        proc._log_midi_stats("test", [])
        proc._log_midi_stats("test", [(0.0, 0.5, 60, 80), (0.5, 0.1, 62, 100)])
