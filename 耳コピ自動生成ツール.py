#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
耳コピ自動生成ツール v3.0
MP3をドラッグ&ドロップするだけで耳コピ音源（MIDI再合成）を自動作成

必要環境: Python 3.8+
初回起動時に依存パッケージを自動インストールします
"""

import os
import sys
import subprocess
import platform

# =====================================================
# 自動インストール（初回のみ）
# =====================================================

def _ensure_packages():
    PACKAGES = [
        ("numpy",       "numpy"),
        ("librosa",     "librosa"),
        ("soundfile",   "soundfile"),
        ("scipy",       "scipy"),
        ("mido",        "mido"),
        ("pydub",       "pydub"),
        ("tkinterdnd2", "tkinterdnd2"),
    ]
    missing = [pip for imp, pip in PACKAGES
               if not _importable(imp)]
    if missing:
        print(f"[初回セットアップ] パッケージをインストールします: {', '.join(missing)}")
        for pkg in missing:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                    stderr=subprocess.DEVNULL
                )
                print(f"  ✓ {pkg}")
            except subprocess.CalledProcessError:
                print(f"  ✗ {pkg} のインストールに失敗しました（後で手動インストールを試してください）")
        print("インストール完了。起動します...\n")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _importable(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


_ensure_packages()

# =====================================================
# 依存インポート
# =====================================================

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from pydub import AudioSegment
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import tempfile
from pathlib import Path

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False


# =====================================================
# 音楽分析エンジン
# =====================================================

SR = 44100
HOP = 512


class EarCopyEngine:
    """音楽分析 → 耳コピ音源生成エンジン"""

    def __init__(self, on_progress=None):
        self._cb = on_progress

    # ---- ロギング ----------------------------------------

    def _log(self, msg, pct=None):
        if self._cb:
            self._cb(msg, pct)
        else:
            print(f"[{pct or '--':>3}%] {msg}")

    # ---- メイン処理 --------------------------------------

    def process(self, input_path: str, output_path: str):
        """
        MP3 → 分析 → 合成 → 出力MP3
        戻り値: (成功フラグ, 出力パスまたはエラー文字列)
        """
        try:
            y, sr = self._load(input_path)
            duration = len(y) / sr

            y_h, y_p = self._hpss(y)
            tempo, beats = self._tempo(y, sr)
            self._log(f"テンポ: {tempo:.1f} BPM", 22)

            mel_events  = self._melody(y_h, sr)
            bass_events = self._bass(y_h, sr)
            chord_events = self._chords(y_h, sr, beats)
            drum_events  = self._drums(y_p, sr)

            audio = self._mix(
                duration, mel_events, bass_events,
                chord_events, drum_events
            )

            self._log("MP3を保存中...", 92)
            self._save_mp3(audio, output_path)

            self._log("MIDIを保存中...", 96)
            midi_path = str(Path(output_path).with_suffix(".mid"))
            self._save_midi(
                tempo, duration,
                mel_events, bass_events, chord_events, drum_events,
                midi_path
            )

            self._log("完了！", 100)
            return True, output_path

        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            self._log(f"エラー: {e}", -1)
            return False, msg

    # ---- 音声読み込み ------------------------------------

    def _load(self, path):
        self._log("音源を読み込み中...", 5)
        y, sr = librosa.load(path, sr=SR, mono=True)
        return y, sr

    # ---- ハーモニック/パーカッション分離 -----------------

    def _hpss(self, y):
        self._log("ハーモニック/パーカッション分離中...", 12)
        yh, yp = librosa.effects.hpss(y, margin=3.0)
        return yh, yp

    # ---- テンポ/ビート ----------------------------------

    def _tempo(self, y, sr):
        self._log("テンポ・リズム解析中...", 18)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
        return float(tempo), beat_times

    # ---- メロディー検出 ----------------------------------

    def _melody(self, yh, sr):
        """(time, hz, duration) のリストを返す"""
        self._log("メロディー解析中 (pyin)...", 28)
        f0, voiced, _ = librosa.pyin(
            yh,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr, hop_length=HOP
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
        return self._f0_to_events(f0, voiced, times, max_gap_hz=8)

    # ---- ベース検出 -------------------------------------

    def _bass(self, yh, sr):
        self._log("ベースライン解析中...", 42)
        nyq = sr / 2
        b, a = butter(4, min(250 / nyq, 0.99), btype="low")
        yb = filtfilt(b, a, yh)
        f0, voiced, _ = librosa.pyin(
            yb,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C3"),
            sr=sr, hop_length=HOP * 2
        )
        times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=HOP * 2
        )
        return self._f0_to_events(f0, voiced, times, max_gap_hz=4)

    def _f0_to_events(self, f0, voiced, times, max_gap_hz=6):
        """連続したピッチフレームをノートイベント(time, hz, dur)に変換"""
        events = []
        i = 0
        while i < len(f0):
            if voiced[i] and f0[i] is not None and not np.isnan(f0[i]):
                start_t = times[i]
                base_hz  = f0[i]
                j = i + 1
                while (j < len(f0) and voiced[j] and
                       f0[j] is not None and not np.isnan(f0[j]) and
                       abs(f0[j] - base_hz) < max_gap_hz):
                    j += 1
                end_t = times[min(j, len(times) - 1)]
                dur = end_t - start_t
                if dur >= 0.06:
                    events.append((start_t, base_hz, min(dur, 4.0)))
                i = j
            else:
                i += 1
        return events

    # ---- コード検出 -------------------------------------

    def _chords(self, yh, sr, beat_times):
        self._log("コード解析中...", 50)
        chroma = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=HOP)
        NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        TEMPLATES = {
            "maj":  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            "min":  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            "dom7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        }
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
        events = []
        for idx, bf in enumerate(beat_frames):
            end = (beat_frames[idx + 1]
                   if idx + 1 < len(beat_frames)
                   else min(bf + 8, chroma.shape[1] - 1))
            if bf >= chroma.shape[1]:
                break
            cv = np.mean(chroma[:, bf: end + 1], axis=1)
            cv = cv / (cv.max() + 1e-9)
            best, chord = -1, ("C", "maj")
            for root in range(12):
                for q, tmpl in TEMPLATES.items():
                    s = float(np.dot(cv, np.roll(tmpl, root)))
                    if s > best:
                        best, chord = s, (NOTES[root], q)
            next_t = (beat_times[idx + 1]
                      if idx + 1 < len(beat_times)
                      else beat_times[idx] + 0.5)
            dur = max(next_t - beat_times[idx] - 0.02, 0.05)
            events.append((beat_times[idx], chord[0], chord[1], dur))
        return events

    # ---- ドラム検出 -------------------------------------

    def _drums(self, yp, sr):
        self._log("ドラム解析中...", 58)
        onset_frames = librosa.onset.onset_detect(
            y=yp, sr=sr, hop_length=HOP, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)
        events = []
        for ot, of_ in zip(onset_times, onset_frames):
            s = max(0, of_ - 2) * HOP
            e = min(of_ + 8, len(yp) // HOP) * HOP
            seg = yp[s:e]
            if len(seg) < 8:
                continue
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1 / sr)
            tot = fft.sum() + 1e-9
            low = fft[freqs < 120].sum() / tot
            hi  = fft[freqs >= 1000].sum() / tot
            if low > 0.45:
                kind = "kick"
            elif hi > 0.35:
                kind = "hihat"
            else:
                kind = "snare"
            events.append((ot, kind))
        return events

    # ---- 音声合成 ----------------------------------------

    def _mix(self, duration, mel, bass, chords, drums):
        self._log("音源を合成中...", 68)
        n = int((duration + 2.0) * SR)
        out = np.zeros(n, dtype=np.float64)

        self._log("  → メロディー合成", 70)
        out += self._synth_notes(mel,   n, self._piano,  gain=0.9)
        self._log("  → ベース合成", 74)
        out += self._synth_notes(bass,  n, self._bass_t, gain=0.7)
        self._log("  → コード合成", 78)
        out += self._synth_chords_audio(chords, n, gain=0.6)
        self._log("  → ドラム合成", 82)
        out += self._synth_drum_audio(drums, n, gain=0.55)

        # ピーク正規化
        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak * 0.88
        return out.astype(np.float32)

    # ---- ノートイベント → サンプル ----------------------

    def _synth_notes(self, events, n, tone_fn, gain=1.0):
        buf = np.zeros(n)
        for (start, hz, dur) in events:
            s0 = int(start * SR)
            tone = tone_fn(hz, dur)
            s1 = s0 + len(tone)
            if s1 <= n:
                buf[s0:s1] += tone * gain
        return buf

    def _piano(self, hz, dur):
        t = np.arange(int(dur * SR)) / SR
        sig = np.zeros(len(t))
        for h in range(1, 10):
            hf = hz * h
            if hf >= SR / 2:
                break
            sig += (0.78 ** (h - 1)) * np.sin(2 * np.pi * hf * t)
        env = np.exp(-t * (1.8 + hz / 800))
        env[:min(int(0.005 * SR), len(t))] *= np.linspace(
            0, 1, min(int(0.005 * SR), len(t))
        )
        return (sig * env * 0.28).astype(np.float64)

    def _bass_t(self, hz, dur):
        t = np.arange(int(dur * SR)) / SR
        sig = np.zeros(len(t))
        for h in range(1, 6):
            hf = hz * h
            if hf >= SR / 2:
                break
            sig += (0.85 ** (h - 1)) * np.sin(2 * np.pi * hf * t)
        env = np.exp(-t * 1.6)
        env[:min(int(0.01 * SR), len(t))] *= np.linspace(
            0, 1, min(int(0.01 * SR), len(t))
        )
        return (sig * env * 0.38).astype(np.float64)

    # ---- コード合成 -------------------------------------

    SEMI = 2 ** (1 / 12)
    NOTE_HZ = {
        "C": 130.81, "C#": 138.59, "D": 146.83, "D#": 155.56,
        "E": 164.81, "F": 174.61, "F#": 185.00, "G": 196.00,
        "G#": 207.65, "A": 220.00, "A#": 233.08, "B": 246.94
    }
    CHORD_SEMI = {"maj": [0, 4, 7], "min": [0, 3, 7], "dom7": [0, 4, 7, 10]}

    def _synth_chords_audio(self, events, n, gain=1.0):
        buf = np.zeros(n)
        for (start, root, quality, dur) in events:
            root_hz = self.NOTE_HZ.get(root, 130.81)
            for semi in self.CHORD_SEMI.get(quality, [0, 4, 7]):
                hz = root_hz * (self.SEMI ** semi)
                tone = self._guitar(hz, dur)
                s0 = int(start * SR)
                s1 = s0 + len(tone)
                if s1 <= n:
                    buf[s0:s1] += tone * gain
        return buf

    def _guitar(self, hz, dur):
        t = np.arange(int(dur * SR)) / SR
        sig = np.zeros(len(t))
        for h in range(1, 7):
            hf = hz * h
            if hf >= SR / 2:
                break
            sig += (0.72 ** (h - 1)) * np.sin(2 * np.pi * hf * t)
        env = np.exp(-t * 2.8)
        env[:min(int(0.003 * SR), len(t))] *= np.linspace(
            0, 1, min(int(0.003 * SR), len(t))
        )
        return (sig * env * 0.22).astype(np.float64)

    # ---- ドラム合成 -------------------------------------

    def _synth_drum_audio(self, events, n, gain=1.0):
        buf = np.zeros(n)
        for (start, kind) in events:
            sound = self._drum(kind)
            s0 = int(start * SR)
            s1 = s0 + len(sound)
            if s1 <= n:
                buf[s0:s1] += sound * gain
        return buf

    def _drum(self, kind):
        if kind == "kick":
            t = np.arange(int(0.28 * SR)) / SR
            sweep = 80 * np.exp(-t * 22)
            tone  = np.sin(2 * np.pi * np.cumsum(sweep) / SR)
            noise = 0.25 * np.random.randn(len(t))
            env   = np.exp(-t * 12)
            return ((tone + noise) * env * 0.75).astype(np.float64)

        elif kind == "snare":
            t = np.arange(int(0.14 * SR)) / SR
            tone  = 0.4 * np.sin(2 * np.pi * 195 * t)
            noise = 0.75 * np.random.randn(len(t))
            env   = np.exp(-t * 22)
            return ((tone + noise) * env * 0.60).astype(np.float64)

        else:  # hihat
            t = np.arange(int(0.035 * SR)) / SR
            noise = np.random.randn(len(t))
            nyq = SR / 2
            b, a = butter(4, min(6500 / nyq, 0.99), btype="high")
            noise = filtfilt(b, a, noise)
            return (noise * np.exp(-t * 100) * 0.45).astype(np.float64)

    # ---- ファイル保存 ------------------------------------

    def _save_mp3(self, audio, path):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, audio, SR)
        seg = AudioSegment.from_wav(tmp)
        seg.export(path, format="mp3", bitrate="192k")
        os.unlink(tmp)

    def _save_midi(self, tempo, duration, mel, bass, chords, drums, path):
        mid = MidiFile(type=1, ticks_per_beat=480)
        tpb = 480
        us_per_beat = int(60_000_000 / tempo)

        def abs_to_track(events_abs):
            """(abs_tick, msg) リストをデルタ時刻に変換してトラックへ"""
            track = MidiTrack()
            events_abs.sort(key=lambda x: x[0])
            prev = 0
            for tick, msg in events_abs:
                delta = max(0, tick - prev)
                msg.time = delta
                track.append(msg)
                prev = tick
            return track

        def secs_to_ticks(s):
            return int(s * tempo / 60 * tpb)

        def note_msgs(events, ch, prog):
            evs = [
                (0, MetaMessage("set_tempo", tempo=us_per_beat, time=0)),
                (0, Message("program_change", channel=ch, program=prog, time=0)),
            ]
            for (start, hz, dur) in events:
                note = int(np.clip(librosa.hz_to_midi(hz), 0, 127))
                t0 = secs_to_ticks(start)
                t1 = secs_to_ticks(start + dur)
                evs.append((t0, Message("note_on",  channel=ch, note=note, velocity=88, time=0)))
                evs.append((t1, Message("note_off", channel=ch, note=note, velocity=0,  time=0)))
            return abs_to_track(evs)

        # トラック0: テンポ
        tempo_track = MidiTrack()
        tempo_track.append(MetaMessage("set_tempo", tempo=us_per_beat, time=0))
        mid.tracks.append(tempo_track)

        # メロディー (ch0, Piano)
        mid.tracks.append(note_msgs(mel, 0, 0))
        # ベース (ch1, Acoustic Bass)
        mid.tracks.append(note_msgs(bass, 1, 32))

        # コード (ch2, Acoustic Guitar)
        chord_evs = [
            (0, MetaMessage("set_tempo", tempo=us_per_beat, time=0)),
            (0, Message("program_change", channel=2, program=25, time=0)),
        ]
        SEMI = 2 ** (1 / 12)
        CHORD_SEMI = {"maj": [0, 4, 7], "min": [0, 3, 7], "dom7": [0, 4, 7, 10]}
        NOTE_HZ = self.NOTE_HZ
        for (start, root, quality, dur) in chords:
            root_hz = NOTE_HZ.get(root, 130.81)
            for si in CHORD_SEMI.get(quality, [0, 4, 7]):
                hz  = root_hz * (SEMI ** si)
                note = int(np.clip(librosa.hz_to_midi(hz), 0, 127))
                t0  = secs_to_ticks(start)
                t1  = secs_to_ticks(start + dur)
                chord_evs.append((t0, Message("note_on",  channel=2, note=note, velocity=70, time=0)))
                chord_evs.append((t1, Message("note_off", channel=2, note=note, velocity=0,  time=0)))
        mid.tracks.append(abs_to_track(chord_evs))

        # ドラム (ch9)
        DRUM_MAP = {"kick": 36, "snare": 38, "hihat": 42}
        drum_evs = [
            (0, MetaMessage("set_tempo", tempo=us_per_beat, time=0)),
        ]
        for (start, kind) in drums:
            note = DRUM_MAP.get(kind, 38)
            t0   = secs_to_ticks(start)
            drum_evs.append((t0, Message("note_on",  channel=9, note=note, velocity=95, time=0)))
            drum_evs.append((t0 + 30, Message("note_off", channel=9, note=note, velocity=0, time=0)))
        mid.tracks.append(abs_to_track(drum_evs))

        mid.save(path)


# =====================================================
# GUI
# =====================================================

class App:
    BG      = "#1a1a2e"
    BG2     = "#16213e"
    BG3     = "#0d0d1a"
    ACCENT  = "#e94560"
    FG      = "#e0e0e0"
    FG2     = "#a0a0b0"
    GREEN   = "#4ade80"
    BLUE    = "#4a90d9"

    def __init__(self):
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("耳コピ自動生成ツール")
        self.root.geometry("620x520")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self._out_dir  = tk.StringVar(value=str(Path.home() / "Desktop"))
        self._status   = tk.StringVar(value="MP3ファイルをドロップしてください")
        self._progress = tk.DoubleVar(value=0)
        self._busy     = False

        self._build()

    # ---- UI構築 -----------------------------------------

    def _build(self):
        r = self.root

        # タイトル
        tk.Label(r, text="耳コピ自動生成ツール",
                 font=("Helvetica", 20, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(pady=(20, 4))
        tk.Label(r, text="MP3をAIが分析し、全パートを自動トランスクリプション → 再合成",
                 font=("Helvetica", 9), bg=self.BG, fg=self.FG2).pack()

        # ドロップゾーン
        self._drop_frame = tk.Frame(r, bg=self.BG2, relief="flat", bd=0)
        self._drop_frame.pack(padx=30, pady=(14, 0), fill="x", ipady=28)
        self._drop_lbl = tk.Label(
            self._drop_frame,
            text="ここにMP3をドラッグ & ドロップ\nまたはクリックしてファイル選択",
            font=("Helvetica", 13), bg=self.BG2, fg=self.BLUE,
            cursor="hand2", justify="center"
        )
        self._drop_lbl.pack(expand=True, fill="both", pady=16)

        for w in (self._drop_frame, self._drop_lbl):
            w.bind("<Button-1>", self._click)

        if HAS_DND:
            self._drop_lbl.drop_target_register(DND_FILES)
            self._drop_lbl.dnd_bind("<<Drop>>",      self._drop)
            self._drop_lbl.dnd_bind("<<DragEnter>>", lambda e: self._hover(True))
            self._drop_lbl.dnd_bind("<<DragLeave>>", lambda e: self._hover(False))

        # 出力先
        of = tk.Frame(r, bg=self.BG)
        of.pack(padx=30, pady=(12, 0), fill="x")
        tk.Label(of, text="出力先:", bg=self.BG, fg=self.FG2,
                 font=("Helvetica", 10)).pack(side="left")
        tk.Entry(of, textvariable=self._out_dir,
                 bg=self.BG2, fg=self.FG,
                 insertbackground="white", font=("Helvetica", 9),
                 width=42, relief="flat").pack(side="left", padx=(8, 6))
        tk.Button(of, text="参照", command=self._browse,
                  bg="#0f3460", fg=self.FG, relief="flat",
                  padx=8, pady=3, cursor="hand2").pack(side="left")

        # プログレスバー
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Ear.Horizontal.TProgressbar",
                        troughcolor=self.BG3,
                        background=self.ACCENT,
                        thickness=10)
        self._bar = ttk.Progressbar(r, variable=self._progress,
                                    maximum=100, length=560,
                                    style="Ear.Horizontal.TProgressbar")
        self._bar.pack(pady=(18, 4))

        tk.Label(r, textvariable=self._status,
                 bg=self.BG, fg=self.GREEN,
                 font=("Helvetica", 10)).pack()

        # ログ
        lf = tk.Frame(r, bg=self.BG3)
        lf.pack(padx=30, pady=(10, 20), fill="both", expand=True)
        self._log = tk.Text(
            lf, height=7, bg=self.BG3, fg="#6ee7b7",
            font=("Courier", 8), relief="flat",
            state="disabled", wrap="word"
        )
        self._log.pack(fill="both", expand=True, padx=6, pady=6)

    # ---- イベントハンドラ --------------------------------

    def _hover(self, on):
        c = "#1e3a5f" if on else self.BG2
        self._drop_frame.configure(bg=c)
        self._drop_lbl.configure(bg=c)

    def _browse(self):
        d = filedialog.askdirectory()
        if d:
            self._out_dir.set(d)

    def _click(self, _=None):
        if self._busy:
            return
        files = filedialog.askopenfilenames(
            filetypes=[("音楽ファイル", "*.mp3 *.wav *.m4a *.flac"), ("全て", "*")]
        )
        for f in files:
            self._run(f)

    def _drop(self, event):
        if self._busy:
            return
        for f in self.root.tk.splitlist(event.data):
            if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
                self._run(f)

    # ---- 処理実行 ----------------------------------------

    def _run(self, path):
        if self._busy:
            messagebox.showwarning("処理中", "前の処理が完了してからお試しください。")
            return
        self._busy = True
        self._progress.set(0)

        stem   = Path(path).stem
        output = str(Path(self._out_dir.get()) / f"耳コピ_{stem}.mp3")
        self._append_log(f"入力: {Path(path).name}")
        self._append_log(f"出力: {output}")

        def worker():
            engine = EarCopyEngine(on_progress=self._on_prog)
            ok, res = engine.process(path, output)
            self.root.after(0, lambda: (self._done(res) if ok else self._err(res)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_prog(self, msg, pct):
        self._status.set(msg)
        if pct and pct > 0:
            self._progress.set(pct)
        self._append_log(msg)
        self.root.update_idletasks()

    def _append_log(self, msg):
        self._log.configure(state="normal")
        self._log.insert("end", f"> {msg}\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _done(self, output):
        self._busy = False
        self._status.set("完了！")
        self._progress.set(100)
        self._append_log(f"保存完了: {output}")
        open_dir = messagebox.askyesno(
            "完了",
            f"耳コピ音源を生成しました！\n\n{output}\n\n出力フォルダを開きますか？"
        )
        if open_dir:
            folder = str(Path(output).parent)
            if platform.system() == "Windows":
                os.startfile(folder)
            elif platform.system() == "Darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])

    def _err(self, msg):
        self._busy = False
        self._status.set("エラーが発生しました")
        self._append_log("エラーが発生しました")
        messagebox.showerror("エラー", f"処理に失敗しました:\n\n{msg[:600]}")

    def run(self):
        self.root.mainloop()


# =====================================================
# エントリーポイント
# =====================================================

if __name__ == "__main__":
    try:
        App().run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nエラーが発生しました。上のメッセージをコピーしてください。\nEnterキーで終了...")
