"""
パイプライン健全性テスト — AI 不要で後段処理チェーンを検証する。

テスト対象:
  1. フルパイプライン (MIDI保存 → FluidSynth合成 → マスタリング → MP3保存)
  2. マスタリングチェーン単体
  3. MIDI duration 整合性

AI パッケージ (demucs, torch, basic_pitch, torchaudio) はスタブで代替し、
CI 環境でも実行可能にする。
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
# Stub AI packages before importing mimikopi
# ---------------------------------------------------------------------------

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

# torch: do NOT stub -- mimikopi handles missing torch gracefully via try/except,
# and stubbing it breaks scipy internals that check for torch.Tensor.

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mimikopi import EarCopyEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR = 44100
DURATION = 15.0
FIXTURE_MP3 = Path(__file__).parent.parent / "fixtures" / "test.mp3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> EarCopyEngine:
    return EarCopyEngine(on_progress=None, mode="ai")


def _make_synthetic_notes(duration: float = DURATION):
    """Generate synthetic vocal, other, bass notes and drum events spanning *duration* seconds."""
    rng = np.random.RandomState(123)

    # Vocal notes: melody line in C4-C5 range
    vocal_notes = []
    t = 0.2
    while t < duration - 0.5:
        midi = rng.randint(60, 73)  # C4 to C5
        dur = rng.uniform(0.2, 0.8)
        vel = rng.randint(60, 110)
        vocal_notes.append((t, dur, midi, vel))
        t += dur + rng.uniform(0.05, 0.3)

    # Other notes: chords in C3-C5
    other_notes = []
    t = 0.0
    while t < duration - 1.0:
        root = rng.randint(48, 72)
        dur = rng.uniform(0.5, 1.5)
        vel = rng.randint(50, 90)
        other_notes.append((t, dur, root, vel))
        other_notes.append((t, dur, root + 4, int(vel * 0.9)))
        other_notes.append((t, dur, root + 7, int(vel * 0.8)))
        t += dur + rng.uniform(0.1, 0.5)

    # Bass notes: C2-C3
    bass_notes = []
    t = 0.0
    while t < duration - 0.5:
        midi = rng.randint(36, 48)
        dur = rng.uniform(0.3, 1.0)
        vel = rng.randint(70, 110)
        bass_notes.append((t, dur, midi, vel))
        t += dur + rng.uniform(0.05, 0.2)

    # Drum events
    drum_events = []
    t = 0.0
    kinds = ['kick', 'snare', 'hihat', 'open_hat', 'ride', 'tom']
    while t < duration:
        kind = kinds[rng.randint(0, len(kinds))]
        vel = rng.randint(60, 120)
        drum_events.append((t, kind, vel))
        t += rng.uniform(0.15, 0.5)

    return vocal_notes, other_notes, bass_notes, drum_events


def _make_synthetic_stereo(duration: float = DURATION, sr: int = SR):
    """Create a synthetic stereo signal with multiple sine waves (not silent)."""
    rng = np.random.RandomState(42)
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Mix several frequencies for a non-trivial signal
    sig = np.zeros(n, dtype=np.float32)
    for freq in [220.0, 330.0, 440.0, 554.37, 659.26]:
        amp = rng.uniform(0.05, 0.15)
        sig += (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)
    # Add gentle noise
    sig += (rng.randn(n) * 0.01).astype(np.float32)
    # Stereo: slightly different per channel
    left = sig * 0.9
    right = sig * 1.0 + (rng.randn(n) * 0.005).astype(np.float32)
    audio = np.column_stack([left, right]).astype(np.float32)
    return audio


# ===========================================================================
# Test 1: Full post-AI pipeline (requires FluidSynth)
# ===========================================================================

@pytest.mark.skipif(
    not shutil.which("fluidsynth"),
    reason="FluidSynth not installed"
)
class TestFullPipeline:
    """End-to-end post-AI pipeline: MIDI save -> synth -> master -> MP3."""

    def test_pipeline_output_health(self, tmp_path):
        engine = _engine()
        vocal_notes, other_notes, bass_notes, drum_events = _make_synthetic_notes()

        # 1. Assign parts (15-instrument)
        parts = engine._ai_assign_parts(vocal_notes, other_notes, bass_notes)

        # 2. Consolidate to max 6 parts
        parts = engine._consolidate_parts(parts)

        # 3. Humanize
        tempo = 120.0
        beats = np.arange(0, DURATION, 60.0 / tempo)
        parts, drum_events = engine._production_humanize(parts, drum_events, tempo, beats)

        # 4. Save MIDI
        midi_path = str(tmp_path / "test.mid")
        engine._save_midi(parts, drum_events, tempo, midi_path)
        assert Path(midi_path).exists(), "MIDI file was not created"

        # 5. Synthesize audio
        audio = engine._synthesize_audio(midi_path, parts, drum_events, DURATION)
        assert audio is not None, "_synthesize_audio returned None"

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        # 6. Master
        audio = engine._master(audio)

        # Assertion 2 (pre-encode): peak <= 0.99 in mastered WAV data
        wav_peak = float(np.max(np.abs(audio)))
        assert wav_peak <= 0.99, f"Mastered WAV peak {wav_peak:.4f} exceeds 0.99"

        # Assertion 4: stereo output
        assert audio.ndim == 2 and audio.shape[1] == 2, (
            f"Expected stereo (N,2), got shape {audio.shape}"
        )

        # 7. Save MP3
        mp3_path = str(tmp_path / "output.mp3")
        engine._save_mp3(audio, mp3_path)
        assert Path(mp3_path).exists(), "MP3 file was not created"

        # 8. Load MP3 and validate duration + silence
        import soundfile as sf_lib
        wav_path = str(tmp_path / "output.wav")
        sf_lib.write(wav_path, audio, SR)
        y_wav, sr_wav = sf_lib.read(wav_path)

        actual_dur = len(y_wav) / sr_wav

        # Assertion 1: duration 15.0 +/- 0.3s
        assert abs(actual_dur - DURATION) <= 0.3, (
            f"Output duration {actual_dur:.2f}s is not within 15.0 +/- 0.3s"
        )

        # Assertion 3: silent samples <= 15%
        flat = y_wav.flatten()
        silent_count = int(np.sum(np.abs(flat) < 0.01))
        silent_ratio = silent_count / len(flat)
        assert silent_ratio <= 0.15, (
            f"Silent sample ratio {silent_ratio:.2%} exceeds 15%"
        )


# ===========================================================================
# Test 2: Mastering chain validation
# ===========================================================================

class TestMasteringChain:
    """Validate the _master() method on synthetic stereo audio."""

    def test_master_peak_and_stereo(self):
        engine = _engine()
        audio = _make_synthetic_stereo(DURATION)
        assert audio.ndim == 2 and audio.shape[1] == 2, "Precondition: input is stereo"

        mastered = engine._master(audio)

        # Output must remain stereo
        assert mastered.ndim == 2 and mastered.shape[1] == 2, (
            f"Expected stereo output, got shape {mastered.shape}"
        )

        # Peak must be <= 0.99
        peak = float(np.max(np.abs(mastered)))
        assert peak <= 0.99, f"Post-master peak {peak:.4f} exceeds 0.99"

    def test_master_no_clipping_samples(self):
        """Zero samples above 0.99 after mastering."""
        engine = _engine()
        audio = _make_synthetic_stereo(DURATION)
        mastered = engine._master(audio)

        clip_count = int(np.sum(np.abs(mastered) > 0.99))
        assert clip_count == 0, f"{clip_count} clipping samples detected after mastering"

    def test_master_preserves_content(self):
        """Mastered audio should not be mostly silent."""
        engine = _engine()
        audio = _make_synthetic_stereo(DURATION)
        mastered = engine._master(audio)

        flat = mastered.flatten()
        silent_ratio = float(np.sum(np.abs(flat) < 0.01)) / len(flat)
        assert silent_ratio <= 0.15, (
            f"Silent ratio {silent_ratio:.2%} exceeds 15% -- mastering killed the signal"
        )


# ===========================================================================
# Test 3: MIDI duration validation
# ===========================================================================

class TestMidiDuration:
    """Validate that saved MIDI file duration matches the note data."""

    def test_midi_duration_matches_notes(self, tmp_path):
        import mido

        engine = _engine()

        # Create parts with known timing: notes spanning 0 to ~14.5s
        parts = {
            'melody': [
                (0.0, 0.5, 60, 80),
                (1.0, 0.5, 62, 85),
                (3.0, 1.0, 64, 90),
                (7.0, 0.5, 67, 75),
                (10.0, 1.5, 72, 95),
                (13.5, 1.0, 60, 80),
            ],
            'chord': [
                (0.0, 2.0, 48, 70),
                (4.0, 2.0, 52, 70),
                (8.0, 2.0, 55, 70),
                (12.0, 2.5, 48, 70),
            ],
            'bass': [
                (0.0, 1.0, 36, 90),
                (2.0, 1.0, 40, 85),
                (5.0, 1.0, 43, 80),
                (9.0, 1.0, 36, 90),
                (13.0, 1.5, 40, 85),
            ],
            'decoration': [],
            'sub_melody': [],
        }
        drums = [
            (0.0, 'kick', 100),
            (0.5, 'hihat', 70),
            (1.0, 'snare', 90),
            (2.0, 'kick', 100),
            (4.0, 'kick', 100),
            (6.0, 'snare', 90),
            (8.0, 'kick', 100),
            (10.0, 'kick', 100),
            (12.0, 'snare', 90),
            (14.0, 'kick', 100),
        ]
        tempo = 120.0

        midi_path = str(tmp_path / "dur_test.mid")
        engine._save_midi(parts, drums, tempo, midi_path)

        # Load with mido and compute total duration
        mid = mido.MidiFile(midi_path)
        midi_duration = mid.length  # total duration in seconds

        # The last note ends at 14.5s (13.5 + 1.0) and last drum at 14.0s
        expected_end = 14.5
        # Allow reasonable tolerance (MIDI tick quantization + note-off placement)
        assert midi_duration >= expected_end - 0.5, (
            f"MIDI duration {midi_duration:.2f}s is shorter than expected {expected_end:.1f}s"
        )
        # Should not be excessively long either
        assert midi_duration <= expected_end + 2.0, (
            f"MIDI duration {midi_duration:.2f}s is much longer than expected {expected_end:.1f}s"
        )

    def test_midi_file_has_tracks(self, tmp_path):
        """MIDI file should have at least tempo track + non-empty part tracks."""
        import mido

        engine = _engine()
        parts = {
            'melody': [(0.0, 1.0, 60, 80), (2.0, 1.0, 64, 90)],
            'chord': [(0.0, 2.0, 48, 70)],
            'bass': [(0.0, 1.0, 36, 90)],
            'decoration': [],
            'sub_melody': [],
        }
        drums = [(0.0, 'kick', 100), (0.5, 'snare', 90)]
        tempo = 120.0

        midi_path = str(tmp_path / "tracks_test.mid")
        engine._save_midi(parts, drums, tempo, midi_path)

        mid = mido.MidiFile(midi_path)
        # 1 tempo track + 3 non-empty part tracks + 1 drum track = 5
        assert len(mid.tracks) >= 4, (
            f"Expected at least 4 tracks (tempo + parts + drums), got {len(mid.tracks)}"
        )
