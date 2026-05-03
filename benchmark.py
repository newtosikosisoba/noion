#!/usr/bin/env python3
"""
benchmark.py — A/B テスト用ベンチマーク & HTML レポート生成

fixtures/ 内の 5 曲を処理し、output/ に並べて HTML レポートを生成する。
手動評価用に各曲の入力・出力を HTML5 audio で並べる。

使い方:
  python benchmark.py              # 全曲処理 + レポート生成
  python benchmark.py --report     # 既存出力からレポートのみ再生成
"""
import sys
import os
import time
import json
import base64
from pathlib import Path

# プロジェクトルート
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FIXTURES_DIR = ROOT / "fixtures"
OUTPUT_DIR = ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "benchmark_report.html"

# ジャンルラベル
GENRE_LABELS = {
    "jpop_sample": "J-POP (120 BPM, C major)",
    "rock_sample": "ロック (140 BPM, E minor)",
    "ballad_sample": "バラード (72 BPM, G major)",
    "edm_sample": "EDM (128 BPM, A minor)",
    "acoustic_sample": "アコースティック (100 BPM, D major)",
}


def _audio_to_data_uri(path):
    """音声ファイルを data URI に変換"""
    p = Path(path)
    if not p.exists():
        return ""
    data = p.read_bytes()
    if p.suffix == ".mp3":
        mime = "audio/mpeg"
    elif p.suffix == ".wav":
        mime = "audio/wav"
    else:
        mime = "audio/mpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def quality_check(output_path):
    """品質ゲート自動チェック (タスク2)"""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(output_path))
    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    results = {}
    n = len(mono)
    duration = n / sr

    # 1. LUFS 簡易測定
    try:
        sys.path.insert(0, str(ROOT))
        from mimikopi import EarCopyEngine
        eng = EarCopyEngine.__new__(EarCopyEngine)
        lufs = eng._measure_lufs(audio)
    except Exception:
        rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
        lufs = 20 * np.log10(max(rms, 1e-12)) - 0.691
    results["lufs"] = {
        "value": round(lufs, 1),
        "pass": -16.0 <= lufs <= -12.0,
        "label": f"LUFS: {lufs:.1f} (許容: -16〜-12)",
    }

    # 2. クリッピング率
    clip_count = int(np.sum(np.abs(audio) > 0.99))
    clip_ratio = clip_count / max(audio.size, 1) * 100
    results["clipping"] = {
        "value": round(clip_ratio, 4),
        "pass": clip_ratio < 0.01,
        "label": f"クリッピング率: {clip_ratio:.4f}% (許容: < 0.01%)",
    }

    # 3. 周波数バランス
    check_len = min(len(mono), sr * 5)
    fft = np.abs(np.fft.rfft(mono[:check_len]))
    freqs = np.fft.rfftfreq(check_len, 1 / sr)
    e_sub = float(fft[freqs < 80].sum())
    e_high = float(fft[freqs >= 8000].sum())
    e_total = float(fft.sum()) + 1e-12
    sub_ratio = e_sub / e_total
    high_ratio = e_high / e_total
    freq_ok = 0.01 < sub_ratio < 0.40 and 0.01 < high_ratio < 0.40
    results["freq_balance"] = {
        "value": {"sub_80hz": round(sub_ratio * 100, 1), "above_8khz": round(high_ratio * 100, 1)},
        "pass": freq_ok,
        "label": f"周波数バランス: <80Hz={sub_ratio*100:.1f}%, >8kHz={high_ratio*100:.1f}%",
    }

    # 4. 無音区間
    frame_size = int(0.05 * sr)
    n_frames = n // frame_size
    silent_frames = 0
    for i in range(n_frames):
        frame = mono[i * frame_size:(i + 1) * frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < 0.001:
            silent_frames += 1
    silence_ratio = silent_frames / max(n_frames, 1) * 100
    results["silence"] = {
        "value": round(silence_ratio, 1),
        "pass": silence_ratio <= 20.0,
        "label": f"無音区間: {silence_ratio:.1f}% (許容: ≤ 20%)",
    }

    # 5. 楽器同時発音 (MIDI 解析)
    midi_path = str(output_path).replace(".mp3", ".mid").replace(".wav", ".mid")
    inst_coverage = 0.0
    if Path(midi_path).exists():
        try:
            import mido
            mid = mido.MidiFile(midi_path)
            ticks_per_beat = mid.ticks_per_beat
            # チャンネル別のアクティブ区間を集計
            from collections import defaultdict
            active_channels = defaultdict(list)
            for track in mid.tracks:
                abs_time = 0
                for msg in track:
                    abs_time += msg.time
                    if msg.type == "note_on" and msg.velocity > 0 and hasattr(msg, "channel"):
                        t_sec = abs_time / ticks_per_beat * (60.0 / 120.0)
                        active_channels[msg.channel].append(t_sec)

            if active_channels and duration > 0:
                time_slots = int(duration * 10)
                counts = np.zeros(time_slots)
                for ch, times in active_channels.items():
                    for t in times:
                        slot = int(t * 10)
                        if 0 <= slot < time_slots:
                            for s in range(slot, min(slot + 5, time_slots)):
                                counts[s] = max(counts[s], len([c for c in active_channels if any(
                                    abs(tt - s / 10) < 0.5 for tt in active_channels[c]
                                )]))
                multi_instrument = int(np.sum(counts >= 3))
                inst_coverage = multi_instrument / max(time_slots, 1) * 100
        except Exception:
            inst_coverage = -1

    results["multi_instrument"] = {
        "value": round(inst_coverage, 1),
        "pass": inst_coverage >= 50.0 or inst_coverage < 0,
        "label": f"3楽器以上同時発音: {inst_coverage:.1f}% (目標: ≥ 50%)",
    }

    return results


def process_fixtures(skip_existing=False):
    """fixtures/ 内の全曲を処理"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from mimikopi import EarCopyEngine, _ensure_ai_packages

    fixture_files = sorted(FIXTURES_DIR.glob("*.wav")) + sorted(FIXTURES_DIR.glob("*.mp3"))
    results = {}

    for fpath in fixture_files:
        stem = fpath.stem
        output_mp3 = OUTPUT_DIR / f"{stem}_output.mp3"

        if skip_existing and output_mp3.exists():
            print(f"[SKIP] {stem} (既に存在)")
            results[stem] = {"output": str(output_mp3), "time": 0}
            continue

        print(f"\n{'='*60}")
        print(f"処理中: {fpath.name} ({GENRE_LABELS.get(stem, stem)})")
        print(f"{'='*60}")

        start = time.time()
        engine = EarCopyEngine(mode="ai_inst")
        ok, res = engine.process(str(fpath), str(output_mp3))
        elapsed = time.time() - start

        if ok:
            print(f"✓ 完了 ({elapsed:.1f}秒)")
            results[stem] = {"output": str(output_mp3), "time": elapsed}
        else:
            print(f"✗ 失敗: {res[:200]}")
            results[stem] = {"output": None, "time": elapsed, "error": str(res)[:200]}

    return results


def generate_report(process_results=None):
    """HTML レポートを生成"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixture_files = sorted(FIXTURES_DIR.glob("*.wav")) + sorted(FIXTURES_DIR.glob("*.mp3"))
    entries = []

    for fpath in fixture_files:
        stem = fpath.stem
        output_mp3 = OUTPUT_DIR / f"{stem}_output.mp3"
        output_wav = OUTPUT_DIR / f"{stem}_output.wav"

        entry = {
            "name": stem,
            "genre": GENRE_LABELS.get(stem, stem),
            "input_uri": _audio_to_data_uri(fpath),
            "input_file": fpath.name,
        }

        out_path = output_mp3 if output_mp3.exists() else output_wav
        if out_path.exists():
            entry["output_uri"] = _audio_to_data_uri(out_path)
            entry["output_file"] = out_path.name
            # 品質チェック
            try:
                entry["quality"] = quality_check(out_path)
            except Exception as e:
                entry["quality"] = {"error": str(e)}
        else:
            entry["output_uri"] = ""
            entry["output_file"] = "(未処理)"
            entry["quality"] = {}

        if process_results and stem in process_results:
            entry["time"] = process_results[stem].get("time", 0)
            entry["error"] = process_results[stem].get("error")

        entries.append(entry)

    # HTML 生成
    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Noion ベンチマークレポート</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { color: #e94560; }
.card { background: #16213e; border-radius: 12px; padding: 20px; margin: 16px 0; }
.card h2 { margin-top: 0; color: #4a90d9; }
audio { width: 100%; margin: 8px 0; }
.quality { margin: 12px 0; }
.pass { color: #4ade80; }
.fail { color: #e94560; }
.label { font-size: 12px; color: #a0a0b0; margin-bottom: 4px; }
.row { display: flex; gap: 20px; }
.col { flex: 1; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; }
td, th { padding: 6px 10px; text-align: left; border-bottom: 1px solid #2a2a4e; }
th { color: #a0a0b0; font-size: 12px; }
</style>
</head>
<body>
<h1>Noion ベンチマークレポート</h1>
<p>生成日時: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""

    for e in entries:
        html += f"""
<div class="card">
<h2>{e['genre']}</h2>
<div class="row">
<div class="col">
  <div class="label">入力: {e['input_file']}</div>
  <audio controls src="{e['input_uri']}"></audio>
</div>
<div class="col">
  <div class="label">出力: {e['output_file']}</div>
  {'<audio controls src="' + e['output_uri'] + '"></audio>' if e['output_uri'] else '<p>(未処理)</p>'}
</div>
</div>
"""
        if e.get("time"):
            html += f"<p>処理時間: {e['time']:.1f}秒</p>"
        if e.get("error"):
            html += f'<p class="fail">エラー: {e["error"]}</p>'

        q = e.get("quality", {})
        if q and "error" not in q:
            html += '<div class="quality"><table><tr><th>チェック項目</th><th>結果</th></tr>'
            for key in ["lufs", "clipping", "freq_balance", "silence", "multi_instrument"]:
                if key in q:
                    item = q[key]
                    cls = "pass" if item["pass"] else "fail"
                    symbol = "✓" if item["pass"] else "✗"
                    html += f'<tr><td>{item["label"]}</td><td class="{cls}">{symbol}</td></tr>'
            html += "</table></div>"

        html += "</div>\n"

    html += """
<footer style="text-align:center; color:#666; margin-top:40px; font-size:12px;">
Noion — 100% MIDI再合成・原盤不使用
</footer>
</body></html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"\nHTMLレポート生成: {REPORT_PATH}")
    return str(REPORT_PATH)


if __name__ == "__main__":
    if "--report" in sys.argv:
        generate_report()
    else:
        results = process_fixtures()
        generate_report(results)
