#!/usr/bin/env python3
"""
品質計測 CLI — 入力 MP3 と出力 MP3 を比較し 10 項目を表形式で表示する。

使い方:
    python scripts/measure_quality.py input.mp3 output.mp3
    python scripts/measure_quality.py input.mp3 output.mp3 --midi output.mid
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 44100

# ── ターゲット定義 ──────────────────────────────────────────

TARGETS = {
    "T1  Duration (s)":        {"min": 14.7, "max": 15.3},
    "T2  Peak":                {"min": 0.0,  "max": 0.95},
    "T3  Clipped samples":     {"min": 0,    "max": 0},
    "T4  Silence (%)":         {"min": 5.0,  "max": 15.0},
    "T5  RMS":                 {"min": 0.18, "max": 0.32},
    "T6a Sub (<60Hz)":         {"min": 0.7,  "max": 1.5},
    "T6b Bass (60-250Hz)":     {"min": 0.7,  "max": 1.4},
    "T6c LMid (250-800Hz)":    {"min": 0.7,  "max": 1.5},
    "T6d Mid (800-2kHz)":      {"min": 0.7,  "max": 1.4},
    "T6e HMid (2k-8kHz)":     {"min": 0.6,  "max": 1.4},
    "T6f Air (>8kHz)":         {"min": 0.5,  "max": 1.3},
    "T7  Onset ratio":         {"min": 0.85, "max": 999.0},
    "T8  BPM diff":            {"min": 0.0,  "max": 8.0},
    "T9a Vel melody":          {"min": 70,   "max": 127},
    "T9b Vel chord":           {"min": 65,   "max": 127},
    "T9c Vel bass":            {"min": 85,   "max": 127},
    "T9d Vel pad mean":        {"min": 55,   "max": 127},
    "T9e Vel pad std":         {"min": 5.0,  "max": 999.0},
    "T10 Stereo corr":         {"min": -1.0, "max": 0.95},
}


def _band_energy(fft, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sum(fft[mask] ** 2)) + 1e-12


def _get_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if np.ndim(tempo) == 0:
        return float(tempo)
    return float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)


def measure(input_path, output_path, midi_path=None):
    """全 10 項目を計測し {key: value} の dict を返す。"""
    y_in, _ = librosa.load(str(input_path), sr=SR, mono=True)
    y_out_mono, _ = librosa.load(str(output_path), sr=SR, mono=True)
    y_out_stereo, _ = librosa.load(str(output_path), sr=SR, mono=False)

    results = {}

    # T1 Duration
    results["T1  Duration (s)"] = len(y_out_mono) / SR

    # T2 Peak
    if y_out_stereo.ndim == 2:
        results["T2  Peak"] = float(np.max(np.abs(y_out_stereo)))
    else:
        results["T2  Peak"] = float(np.max(np.abs(y_out_mono)))

    # T3 Clipped samples
    if y_out_stereo.ndim == 2:
        results["T3  Clipped samples"] = int(np.sum(np.abs(y_out_stereo) >= 1.0))
    else:
        results["T3  Clipped samples"] = int(np.sum(np.abs(y_out_mono) >= 1.0))

    # T4 Silence %
    rms_frames = librosa.feature.rms(y=y_out_mono, frame_length=2048, hop_length=512)[0]
    silence_threshold = 0.01
    silence_pct = float(np.sum(rms_frames < silence_threshold)) / max(len(rms_frames), 1) * 100
    results["T4  Silence (%)"] = silence_pct

    # T5 RMS
    results["T5  RMS"] = float(np.sqrt(np.mean(y_out_mono ** 2)))

    # T6 Frequency balance ratios (output / input)
    n_out = min(len(y_out_mono), SR * 10)
    n_in = min(len(y_in), SR * 10)
    fft_out = np.abs(np.fft.rfft(y_out_mono[:n_out]))
    fft_in = np.abs(np.fft.rfft(y_in[:n_in]))
    freqs_out = np.fft.rfftfreq(n_out, 1 / SR)
    freqs_in = np.fft.rfftfreq(n_in, 1 / SR)

    bands = [
        ("T6a Sub (<60Hz)", 0, 60),
        ("T6b Bass (60-250Hz)", 60, 250),
        ("T6c LMid (250-800Hz)", 250, 800),
        ("T6d Mid (800-2kHz)", 800, 2000),
        ("T6e HMid (2k-8kHz)", 2000, 8000),
        ("T6f Air (>8kHz)", 8000, SR // 2),
    ]
    for name, lo, hi in bands:
        e_out = _band_energy(fft_out, freqs_out, lo, hi)
        e_in = _band_energy(fft_in, freqs_in, lo, hi)
        results[name] = e_out / e_in

    # T7 Onset strength ratio
    onset_in = librosa.onset.onset_strength(y=y_in, sr=SR)
    onset_out = librosa.onset.onset_strength(y=y_out_mono, sr=SR)
    results["T7  Onset ratio"] = float(np.mean(onset_out)) / max(float(np.mean(onset_in)), 1e-9)

    # T8 BPM diff
    bpm_in = _get_bpm(y_in, SR)
    bpm_out = _get_bpm(y_out_mono, SR)
    results["T8  BPM diff"] = abs(bpm_out - bpm_in)

    # T9 Part velocities (requires MIDI)
    MIDI_MAP = {
        'melody': (0, 0), 'chord': (1, 4), 'bass': (2, 33),
        'pad': (3, 89), 'decoration': (4, 9), 'sub_melody': (5, 42),
    }
    if midi_path and Path(midi_path).exists():
        try:
            import mido
            mid = mido.MidiFile(str(midi_path))
            ch_to_part = {}
            for part_name, (ch, _prog) in MIDI_MAP.items():
                ch_to_part[ch] = part_name

            part_vels = {}
            for track in mid.tracks:
                for msg in track:
                    if msg.type == "note_on" and msg.velocity > 0 and hasattr(msg, "channel"):
                        part_name = ch_to_part.get(msg.channel, None)
                        if part_name:
                            part_vels.setdefault(part_name, []).append(msg.velocity)

            for part, key in [("melody", "T9a Vel melody"), ("chord", "T9b Vel chord"),
                              ("bass", "T9c Vel bass")]:
                vels = part_vels.get(part, [])
                results[key] = float(np.mean(vels)) if vels else 0.0

            pad_vels = part_vels.get("pad", [])
            results["T9d Vel pad mean"] = float(np.mean(pad_vels)) if pad_vels else 0.0
            results["T9e Vel pad std"] = float(np.std(pad_vels)) if len(pad_vels) > 1 else 0.0
        except Exception as e:
            for k in ["T9a Vel melody", "T9b Vel chord", "T9c Vel bass",
                       "T9d Vel pad mean", "T9e Vel pad std"]:
                results[k] = None
    else:
        for k in ["T9a Vel melody", "T9b Vel chord", "T9c Vel bass",
                   "T9d Vel pad mean", "T9e Vel pad std"]:
            results[k] = None

    # T10 Stereo correlation
    if y_out_stereo.ndim == 2 and y_out_stereo.shape[0] == 2:
        corr = float(np.corrcoef(y_out_stereo[0], y_out_stereo[1])[0, 1])
        results["T10 Stereo corr"] = corr
    else:
        results["T10 Stereo corr"] = 1.0

    return results


def format_table(results):
    """結果を表形式の文字列にする。"""
    lines = []
    lines.append(f"{'Item':<28} {'Value':>10}  {'Target':>16}  {'Status':>6}")
    lines.append("-" * 66)

    pass_count = 0
    fail_count = 0
    na_count = 0

    for key, target in TARGETS.items():
        val = results.get(key)
        lo, hi = target["min"], target["max"]
        if val is None:
            status = "N/A"
            val_str = "N/A"
            na_count += 1
        else:
            if isinstance(val, int):
                val_str = f"{val}"
            else:
                val_str = f"{val:.4f}"
            if lo <= val <= hi:
                status = "PASS"
                pass_count += 1
            else:
                status = "FAIL"
                fail_count += 1

        if hi >= 999:
            target_str = f">= {lo}"
        elif lo <= 0:
            target_str = f"<= {hi}"
        else:
            target_str = f"[{lo}, {hi}]"

        lines.append(f"{key:<28} {val_str:>10}  {target_str:>16}  {status:>6}")

    lines.append("-" * 66)
    lines.append(f"Result: {pass_count} PASS, {fail_count} FAIL, {na_count} N/A")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="品質計測ツール")
    parser.add_argument("input_mp3", help="入力 (原曲) MP3")
    parser.add_argument("output_mp3", help="出力 MP3")
    parser.add_argument("--midi", help="出力 MIDI (T9 パートベロシティ用)", default=None)
    args = parser.parse_args()

    for p in [args.input_mp3, args.output_mp3]:
        if not Path(p).exists():
            print(f"ERROR: {p} が見つかりません", file=sys.stderr)
            sys.exit(1)

    results = measure(args.input_mp3, args.output_mp3, args.midi)
    print(format_table(results))


if __name__ == "__main__":
    main()
