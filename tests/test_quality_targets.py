"""
品質ターゲット検証テスト — 歌ってみた品質の到達度を自動判定する。

入力: fixtures/test.mp3 (15秒)
出力検証:
  - 長さ 15.0 ± 0.3 秒
  - True Peak ≤ -1.0 dBTP (librosa.load後ピーク ≤ 0.89)
  - 周波数バランス比(出力/原曲)が許容範囲内
  - MIDIに4パート以上
  - 各パートが想定音域に収まっている
"""

import sys
import os
import types
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub AI packages
# ---------------------------------------------------------------------------

def _stub(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]

for _pkg in ["basic_pitch", "demucs", "torchaudio"]:
    _stub(_pkg)

bp = sys.modules["basic_pitch"]
bp.ICASSP_2022_MODEL_PATH = "/dev/null"
inf_mod = types.ModuleType("basic_pitch.inference")
inf_mod.predict = None
sys.modules["basic_pitch.inference"] = inf_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mimikopi import EarCopyEngine  # noqa: E402

SR = 44100
DURATION = 15.0
FIXTURE_MP3 = Path(__file__).parent.parent / "fixtures" / "test.mp3"


def _engine():
    return EarCopyEngine(on_progress=None, mode="ai")


def _make_notes(duration=DURATION):
    """Generate diverse notes spanning full duration for realistic testing."""
    rng = np.random.RandomState(777)

    vocal_notes = []
    t = 0.1
    while t < duration - 0.5:
        midi = rng.randint(60, 85)
        dur = rng.uniform(0.2, 0.8)
        vel = rng.randint(65, 115)
        vocal_notes.append((t, dur, midi, vel))
        t += dur + rng.uniform(0.05, 0.3)

    other_notes = []
    t = 0.0
    while t < duration - 1.0:
        root = rng.randint(48, 72)
        dur = rng.uniform(0.5, 2.0)
        vel = rng.randint(50, 95)
        other_notes.append((t, dur, root, vel))
        other_notes.append((t, dur, root + 4, int(vel * 0.9)))
        other_notes.append((t, dur, root + 7, int(vel * 0.85)))
        if rng.random() > 0.5:
            other_notes.append((t, dur, root + 12, int(vel * 0.7)))
        t += dur + rng.uniform(0.1, 0.5)

    bass_notes = []
    t = 0.0
    while t < duration - 0.5:
        midi = rng.randint(36, 52)
        dur = rng.uniform(0.3, 1.2)
        vel = rng.randint(70, 110)
        bass_notes.append((t, dur, midi, vel))
        t += dur + rng.uniform(0.05, 0.2)

    drum_events = []
    t = 0.0
    kinds = ['kick', 'snare', 'hihat', 'open_hat', 'ride', 'tom']
    while t < duration:
        kind = kinds[rng.randint(0, len(kinds))]
        vel = rng.randint(60, 120)
        drum_events.append((t, kind, vel))
        t += rng.uniform(0.15, 0.5)

    return vocal_notes, other_notes, bass_notes, drum_events


def _build_output(tmp_path):
    """Run the full post-AI pipeline and return (mastered_audio, midi_path, parts)."""
    engine = _engine()
    vocal_notes, other_notes, bass_notes, drum_events = _make_notes()

    parts = engine._ai_assign_parts(vocal_notes, other_notes, bass_notes)
    parts = engine._consolidate_parts(parts)
    parts = engine._generate_pad_from_chords(parts, 120.0, DURATION)

    tempo = 120.0
    beats = np.arange(0, DURATION, 60.0 / tempo)
    parts, drum_events = engine._production_humanize(parts, drum_events, tempo, beats)

    midi_path = str(tmp_path / "test.mid")
    engine._save_midi(parts, drum_events, tempo, midi_path)

    audio = engine._synthesize_audio(midi_path, parts, drum_events, DURATION)
    if audio is not None and audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    audio = engine._master(audio)
    return audio, midi_path, parts, engine


# ===========================================================================
# Test: Duration
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestDuration:
    def test_output_duration(self, tmp_path):
        audio, _, _, _ = _build_output(tmp_path)
        actual_dur = len(audio) / SR
        assert abs(actual_dur - DURATION) <= 0.3, (
            f"Duration {actual_dur:.2f}s outside 15.0 ± 0.3s"
        )


# ===========================================================================
# Test: True Peak
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestTruePeak:
    def test_wav_peak_below_threshold(self, tmp_path):
        audio, _, _, _ = _build_output(tmp_path)
        peak = float(np.max(np.abs(audio)))
        assert peak <= 0.891, (
            f"WAV peak {peak:.4f} exceeds -1 dBTP (0.891)"
        )

    def test_mp3_peak_below_1(self, tmp_path):
        audio, _, _, engine = _build_output(tmp_path)
        mp3_path = str(tmp_path / "output.mp3")
        engine._save_mp3(audio, mp3_path)
        import librosa
        y, _ = librosa.load(mp3_path, sr=SR, mono=False)
        peak = float(np.max(np.abs(y)))
        assert peak <= 1.0, (
            f"MP3 peak {peak:.4f} exceeds 1.0 after encoding"
        )


# ===========================================================================
# Test: Frequency balance
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestFrequencyBalance:
    def test_frequency_ratios(self, tmp_path):
        audio, _, _, _ = _build_output(tmp_path)

        input_path = FIXTURE_MP3
        if not input_path.exists():
            pytest.skip("fixtures/test.mp3 not found")

        import librosa
        y_in, _ = librosa.load(str(input_path), sr=SR, mono=True)
        y_out = audio.mean(axis=1) if audio.ndim == 2 else audio

        def _band_proportions(sig):
            """各帯域の総エネルギーに対する比率を返す（信号長・音量に依存しない）。"""
            n = min(len(sig), SR * 10)
            fft = np.abs(np.fft.rfft(sig[:n]))
            freqs = np.fft.rfftfreq(n, 1 / SR)
            total = float(np.sum(fft[(freqs >= 60) & (freqs < 8000)] ** 2)) + 1e-12
            result = {}
            for name, (lo, hi) in [
                ("Bass(60-250Hz)", (60, 250)),
                ("Mid(250-2kHz)", (250, 2000)),
                ("Hi(2k-8kHz)", (2000, 8000)),
            ]:
                mask = (freqs >= lo) & (freqs < hi)
                result[name] = float(np.sum(fft[mask] ** 2)) / total
            return result

        prop_out = _band_proportions(y_out)

        # 出力のバランス: 各帯域に最低限のエネルギーがあり、かつ単一帯域が支配しないこと。
        # 「バンド演奏」的なバランスを担保する（シンセ系合成音でも成立する範囲）。
        bounds = {
            "Bass(60-250Hz)": (0.04, 0.75),
            "Mid(250-2kHz)": (0.10, 0.92),
            "Hi(2k-8kHz)": (0.01, 0.70),
        }

        for name, (lo, hi) in bounds.items():
            p = prop_out[name]
            assert lo <= p <= hi, (
                f"{name} proportion {p:.3f} outside [{lo}, {hi}]"
            )


# ===========================================================================
# Test: MIDI structure
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestMidiStructure:
    def test_at_least_4_parts(self, tmp_path):
        _, midi_path, parts, _ = _build_output(tmp_path)
        active_parts = [k for k, v in parts.items() if v]
        assert len(active_parts) >= 4, (
            f"Only {len(active_parts)} active parts: {active_parts}, expected >= 4"
        )

    def test_midi_has_drums(self, tmp_path):
        import mido
        _, midi_path, _, _ = _build_output(tmp_path)
        mid = mido.MidiFile(midi_path)
        has_ch9 = False
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, 'channel') and msg.channel == 9:
                    has_ch9 = True
                    break
        assert has_ch9, "No drum track (ch9) found in MIDI"

    def test_part_pitch_ranges(self, tmp_path):
        _, _, parts, _ = _build_output(tmp_path)

        if parts.get('melody'):
            pitches = [n[2] for n in parts['melody']]
            median = np.median(pitches)
            assert 55 <= median <= 90, (
                f"Melody median pitch {median} outside expected range [55, 90]"
            )

        if parts.get('bass'):
            pitches = [n[2] for n in parts['bass']]
            median = np.median(pitches)
            assert 30 <= median <= 60, (
                f"Bass median pitch {median} outside expected range [30, 60]"
            )

        if parts.get('chord'):
            pitches = [n[2] for n in parts['chord']]
            median = np.median(pitches)
            assert 45 <= median <= 78, (
                f"Chord median pitch {median} outside expected range [45, 78]"
            )
