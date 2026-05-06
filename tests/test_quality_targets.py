"""
品質ターゲット検証テスト (Phase 3a) — 歌ってみた品質の到達度を自動判定。

入力: fixtures/test.mp3 (15秒)
合成ノートで post-AI パイプラインを end-to-end 実行し、
出力が 10 項目の数値目標に収まっているか検証する。
失敗時は「現在値 vs 目標値」を表で表示。

T1.  長さ:           15.0 ± 0.3 秒
T2.  ピーク:         ≤ 0.95
T3.  クリップ:       0 サンプル
T4.  無音率:         5〜15%
T5.  RMS:            0.18 〜 0.32
T6.  周波数バランス比 (出力/原曲) — 6帯域
T7.  オンセット強度:  原曲の 0.85 倍以上
T8.  BPM検出:        原曲 ±8 BPM 以内
T9.  パート別ベロシティ (melody≥70, chord≥65, bass≥85, pad≥55+std≥5)
T10. ステレオ相関係数: < 0.95
"""

import sys
import os
import types
import shutil
from pathlib import Path

import numpy as np
import pytest
import librosa

# ---------------------------------------------------------------------------
# Stub AI packages (demucs / basic_pitch not available in test env)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine():
    return EarCopyEngine(on_progress=None, mode="ai")


def _make_notes(duration=DURATION):
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


# ---------------------------------------------------------------------------
# Module-scoped fixture: build pipeline output once
# ---------------------------------------------------------------------------

_CACHED_RESULT = None


def _build_output_cached(tmp_dir):
    global _CACHED_RESULT
    if _CACHED_RESULT is not None:
        return _CACHED_RESULT

    engine = _engine()
    vocal_notes, other_notes, bass_notes, drum_events = _make_notes()

    parts = engine._ai_assign_parts(vocal_notes, other_notes, bass_notes)
    parts = engine._consolidate_parts(parts)
    parts = engine._generate_pad_from_chords(parts, 120.0, DURATION)

    tempo = 120.0
    beats = np.arange(0, DURATION, 60.0 / tempo)
    parts = engine._enforce_velocity_floor(parts, seed=42)
    parts, drum_events = engine._production_humanize(parts, drum_events, tempo, beats)

    midi_path = str(Path(tmp_dir) / "test.mid")
    engine._save_midi(parts, drum_events, tempo, midi_path)

    audio = engine._synthesize_audio(midi_path, parts, drum_events, DURATION)
    if audio is not None and audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    audio = engine._master(audio)

    mp3_path = str(Path(tmp_dir) / "output.mp3")
    engine._save_mp3(audio, mp3_path)

    _CACHED_RESULT = {
        "audio": audio,
        "midi_path": midi_path,
        "mp3_path": mp3_path,
        "parts": parts,
        "drum_events": drum_events,
        "engine": engine,
    }
    return _CACHED_RESULT


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("quality")
    return _build_output_cached(str(tmp_dir))


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _band_energy(fft, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sum(fft[mask] ** 2)) + 1e-12


def _get_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if np.ndim(tempo) == 0:
        return float(tempo)
    return float(tempo[0])


def _fmt(label, actual, lo, hi):
    """失敗メッセージ用フォーマット。"""
    if hi >= 999:
        return f"{label}: actual={actual:.4f}, target >= {lo}"
    if lo <= 0:
        return f"{label}: actual={actual:.4f}, target <= {hi}"
    return f"{label}: actual={actual:.4f}, target=[{lo}, {hi}]"


# ===========================================================================
# T1: Duration
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT1Duration:
    def test_duration(self, pipeline):
        audio = pipeline["audio"]
        actual = len(audio) / SR
        assert abs(actual - DURATION) <= 0.3, _fmt("T1 Duration", actual, 14.7, 15.3)


# ===========================================================================
# T2: Peak
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT2Peak:
    def test_peak(self, pipeline):
        peak = float(np.max(np.abs(pipeline["audio"])))
        assert peak <= 0.95, _fmt("T2 Peak", peak, 0, 0.95)


# ===========================================================================
# T3: Clipped samples
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT3Clip:
    def test_no_clipping(self, pipeline):
        clipped = int(np.sum(np.abs(pipeline["audio"]) >= 1.0))
        assert clipped == 0, f"T3 Clip: {clipped} samples clipped (target: 0)"


# ===========================================================================
# T4: Silence ratio
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT4Silence:
    def test_silence_ratio(self, pipeline):
        audio = pipeline["audio"]
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        rms_frames = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
        silence_pct = float(np.sum(rms_frames < 0.01)) / max(len(rms_frames), 1) * 100
        assert 5.0 <= silence_pct <= 15.0, _fmt("T4 Silence%", silence_pct, 5.0, 15.0)


# ===========================================================================
# T5: RMS
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT5Rms:
    def test_rms(self, pipeline):
        audio = pipeline["audio"]
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        rms = float(np.sqrt(np.mean(mono ** 2)))
        assert 0.18 <= rms <= 0.32, _fmt("T5 RMS", rms, 0.18, 0.32)


# ===========================================================================
# T6: Frequency balance ratios (output / input)
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT6FrequencyBalance:
    BANDS = [
        ("T6a Sub (<60Hz)",      0,    60,  0.7, 1.5),
        ("T6b Bass (60-250Hz)",  60,  250,  0.7, 1.4),
        ("T6c LMid (250-800Hz)", 250, 800,  0.7, 1.5),
        ("T6d Mid (800-2kHz)",   800, 2000, 0.7, 1.4),
        ("T6e HMid (2k-8kHz)",  2000,8000, 0.6, 1.4),
        ("T6f Air (>8kHz)",      8000,SR//2,0.5, 1.3),
    ]

    def test_frequency_ratios(self, pipeline):
        if not FIXTURE_MP3.exists():
            pytest.skip("fixtures/test.mp3 not found")

        audio = pipeline["audio"]
        y_out = audio.mean(axis=1) if audio.ndim == 2 else audio
        y_in, _ = librosa.load(str(FIXTURE_MP3), sr=SR, mono=True)

        n_out = min(len(y_out), SR * 10)
        n_in = min(len(y_in), SR * 10)
        fft_out = np.abs(np.fft.rfft(y_out[:n_out]))
        fft_in = np.abs(np.fft.rfft(y_in[:n_in]))
        freqs_out = np.fft.rfftfreq(n_out, 1 / SR)
        freqs_in = np.fft.rfftfreq(n_in, 1 / SR)

        # 入力の全帯域エネルギー (バンドの有意性判定用)
        total_in = float(np.sum(fft_in ** 2)) + 1e-12

        failures = []
        lines = [f"{'Band':<24} {'Ratio':>8}  {'Target':>14}  {'Status':>6}"]
        lines.append("-" * 58)

        for name, lo, hi, r_min, r_max in self.BANDS:
            e_out = _band_energy(fft_out, freqs_out, lo, hi)
            e_in = _band_energy(fft_in, freqs_in, lo, hi)
            # 入力帯域が全エネルギーの 0.5% 未満なら比率は無意味 → スキップ
            if e_in / total_in < 0.005:
                lines.append(f"{name:<24} {'SKIP':>8}  (ref < 0.5%)")
                continue
            ratio = e_out / e_in
            ok = r_min <= ratio <= r_max
            status = "PASS" if ok else "FAIL"
            lines.append(f"{name:<24} {ratio:>8.4f}  [{r_min}, {r_max}]  {status:>6}")
            if not ok:
                failures.append(name)

        table = "\n".join(lines)
        assert not failures, f"Frequency balance failures:\n{table}"


# ===========================================================================
# T7: Onset strength ratio
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT7Onset:
    def test_onset_strength(self, pipeline):
        if not FIXTURE_MP3.exists():
            pytest.skip("fixtures/test.mp3 not found")

        audio = pipeline["audio"]
        y_out = audio.mean(axis=1) if audio.ndim == 2 else audio
        y_in, _ = librosa.load(str(FIXTURE_MP3), sr=SR, mono=True)

        onset_in = float(np.mean(librosa.onset.onset_strength(y=y_in, sr=SR)))
        onset_out = float(np.mean(librosa.onset.onset_strength(y=y_out, sr=SR)))
        ratio = onset_out / max(onset_in, 1e-9)
        assert ratio >= 0.85, _fmt("T7 Onset ratio", ratio, 0.85, 999)


# ===========================================================================
# T8: BPM detection
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT8Bpm:
    def test_bpm_diff(self, pipeline):
        if not FIXTURE_MP3.exists():
            pytest.skip("fixtures/test.mp3 not found")

        audio = pipeline["audio"]
        y_out = audio.mean(axis=1) if audio.ndim == 2 else audio
        y_in, _ = librosa.load(str(FIXTURE_MP3), sr=SR, mono=True)

        bpm_in = _get_bpm(y_in, SR)
        bpm_out = _get_bpm(y_out, SR)
        diff = abs(bpm_out - bpm_in)
        assert diff <= 8.0, (
            f"T8 BPM: input={bpm_in:.1f}, output={bpm_out:.1f}, "
            f"diff={diff:.1f}, target <= 8.0"
        )


# ===========================================================================
# T9: Part velocity averages
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT9Velocity:
    def test_melody_velocity(self, pipeline):
        vels = [n[3] for n in pipeline["parts"].get("melody", [])]
        if not vels:
            pytest.skip("No melody notes")
        avg = float(np.mean(vels))
        assert avg >= 70, _fmt("T9a Vel melody", avg, 70, 127)

    def test_chord_velocity(self, pipeline):
        vels = [n[3] for n in pipeline["parts"].get("chord", [])]
        if not vels:
            pytest.skip("No chord notes")
        avg = float(np.mean(vels))
        assert avg >= 65, _fmt("T9b Vel chord", avg, 65, 127)

    def test_bass_velocity(self, pipeline):
        vels = [n[3] for n in pipeline["parts"].get("bass", [])]
        if not vels:
            pytest.skip("No bass notes")
        avg = float(np.mean(vels))
        assert avg >= 85, _fmt("T9c Vel bass", avg, 85, 127)

    def test_pad_velocity(self, pipeline):
        vels = [n[3] for n in pipeline["parts"].get("pad", [])]
        if not vels:
            pytest.skip("No pad notes")
        avg = float(np.mean(vels))
        assert avg >= 55, _fmt("T9d Vel pad mean", avg, 55, 127)

    def test_pad_velocity_variation(self, pipeline):
        vels = [n[3] for n in pipeline["parts"].get("pad", [])]
        if len(vels) < 2:
            pytest.skip("Not enough pad notes")
        std = float(np.std(vels))
        assert std >= 5.0, _fmt("T9e Vel pad std", std, 5.0, 999)


# ===========================================================================
# T10: Stereo correlation
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestT10Stereo:
    def test_stereo_correlation(self, pipeline):
        audio = pipeline["audio"]
        if audio.ndim != 2 or audio.shape[1] != 2:
            pytest.fail("Output is not stereo")
        corr = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
        assert corr < 0.95, _fmt("T10 Stereo corr", corr, 0, 0.95)


# ===========================================================================
# Summary: all metrics in one table (always prints, fails if any metric fails)
# ===========================================================================

@pytest.mark.skipif(not shutil.which("fluidsynth"), reason="FluidSynth not installed")
class TestSummary:
    def test_quality_summary(self, pipeline):
        """全 T1-T10 を計測し、結果を表形式で出力する。"""
        audio = pipeline["audio"]
        parts = pipeline["parts"]
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio

        results = []

        # T1
        dur = len(audio) / SR
        results.append(("T1  Duration (s)", dur, 14.7, 15.3))

        # T2
        peak = float(np.max(np.abs(audio)))
        results.append(("T2  Peak", peak, 0.0, 0.95))

        # T3
        clipped = int(np.sum(np.abs(audio) >= 1.0))
        results.append(("T3  Clipped", float(clipped), 0, 0))

        # T4
        rms_frames = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
        silence_pct = float(np.sum(rms_frames < 0.01)) / max(len(rms_frames), 1) * 100
        results.append(("T4  Silence (%)", silence_pct, 5.0, 15.0))

        # T5
        rms = float(np.sqrt(np.mean(mono ** 2)))
        results.append(("T5  RMS", rms, 0.18, 0.32))

        # T6
        if FIXTURE_MP3.exists():
            y_in, _ = librosa.load(str(FIXTURE_MP3), sr=SR, mono=True)
            n_out = min(len(mono), SR * 10)
            n_in = min(len(y_in), SR * 10)
            fft_out = np.abs(np.fft.rfft(mono[:n_out]))
            fft_in = np.abs(np.fft.rfft(y_in[:n_in]))
            freqs_out = np.fft.rfftfreq(n_out, 1 / SR)
            freqs_in = np.fft.rfftfreq(n_in, 1 / SR)
            total_in = float(np.sum(fft_in ** 2)) + 1e-12
            for name, lo, hi, r_min, r_max in [
                ("T6a Sub",  0, 60, 0.7, 1.5),
                ("T6b Bass", 60, 250, 0.7, 1.4),
                ("T6c LMid", 250, 800, 0.7, 1.5),
                ("T6d Mid",  800, 2000, 0.7, 1.4),
                ("T6e HMid", 2000, 8000, 0.6, 1.4),
                ("T6f Air",  8000, SR // 2, 0.5, 1.3),
            ]:
                e_out = _band_energy(fft_out, freqs_out, lo, hi)
                e_in = _band_energy(fft_in, freqs_in, lo, hi)
                if e_in / total_in < 0.005:
                    continue
                results.append((name, e_out / e_in, r_min, r_max))

            # T7
            onset_in = float(np.mean(librosa.onset.onset_strength(y=y_in, sr=SR)))
            onset_out = float(np.mean(librosa.onset.onset_strength(y=mono, sr=SR)))
            results.append(("T7  Onset ratio", onset_out / max(onset_in, 1e-9), 0.85, 999))

            # T8
            bpm_in = _get_bpm(y_in, SR)
            bpm_out = _get_bpm(mono, SR)
            results.append(("T8  BPM diff", abs(bpm_out - bpm_in), 0.0, 8.0))

        # T9
        for part, label, lo in [("melody", "T9a melody vel", 70),
                                 ("chord", "T9b chord vel", 65),
                                 ("bass", "T9c bass vel", 85)]:
            vels = [n[3] for n in parts.get(part, [])]
            avg = float(np.mean(vels)) if vels else 0.0
            results.append((label, avg, lo, 127))

        pad_vels = [n[3] for n in parts.get("pad", [])]
        pad_avg = float(np.mean(pad_vels)) if pad_vels else 0.0
        pad_std = float(np.std(pad_vels)) if len(pad_vels) > 1 else 0.0
        results.append(("T9d pad vel mean", pad_avg, 55, 127))
        results.append(("T9e pad vel std", pad_std, 5.0, 999))

        # T10
        if audio.ndim == 2 and audio.shape[1] == 2:
            corr = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
            results.append(("T10 Stereo corr", corr, -1.0, 0.95))

        # ── 表出力 ──
        failures = []
        lines = [
            "",
            "=" * 62,
            "品質ターゲット計測結果",
            "=" * 62,
            f"{'Item':<22} {'Actual':>10}  {'Target':>16}  {'OK?':>4}",
            "-" * 62,
        ]
        for label, val, lo, hi in results:
            if hi >= 999:
                tgt_str = f">= {lo}"
            elif lo <= 0:
                tgt_str = f"<= {hi}"
            else:
                tgt_str = f"[{lo}, {hi}]"
            ok = lo <= val <= hi
            status = "OK" if ok else "NG"
            if isinstance(val, int) or (isinstance(val, float) and val == int(val) and val < 100):
                val_str = f"{int(val)}"
            else:
                val_str = f"{val:.4f}"
            lines.append(f"{label:<22} {val_str:>10}  {tgt_str:>16}  {status:>4}")
            if not ok:
                failures.append(label)

        lines.append("-" * 62)
        pass_n = len(results) - len(failures)
        lines.append(f"Total: {pass_n}/{len(results)} PASS")
        lines.append("=" * 62)
        table = "\n".join(lines)

        print(table)

        assert not failures, f"{len(failures)} targets missed:\n{table}"
