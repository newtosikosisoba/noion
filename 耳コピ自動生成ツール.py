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
import shutil
import json
from pathlib import Path

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False


# =====================================================
# ffmpeg 自動検出・設定
# =====================================================

_FFMPEG_CONFIG = Path(__file__).parent / ".ffmpeg_path.json"

def _find_ffmpeg():
    """ffmpegのパスを解決する。見つからなければユーザーに選択させる"""
    # 1. PATH にある場合はそのまま使う
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")

    # 2. 前回設定を保存していた場合
    if _FFMPEG_CONFIG.exists():
        try:
            saved = json.loads(_FFMPEG_CONFIG.read_text())
            p = saved.get("path", "")
            if p and Path(p).exists():
                return p
        except Exception:
            pass

    # 3. よくある場所を自動検索（Windows）
    candidates = []
    if platform.system() == "Windows":
        for drive in ["C:", "D:"]:
            for name in ["ffmpeg", "ffmpeg-*"]:
                import glob
                candidates += glob.glob(f"{drive}\\{name}\\bin\\ffmpeg.exe")
                candidates += glob.glob(f"{drive}\\Program Files\\{name}\\bin\\ffmpeg.exe")
                candidates += glob.glob(f"{drive}\\Users\\*\\Downloads\\{name}*\\bin\\ffmpeg.exe")
        for c in candidates:
            if Path(c).exists():
                _save_ffmpeg(c)
                return c

    # 4. ユーザーに手動選択させる（GUIダイアログ）
    import tkinter as _tk
    from tkinter import filedialog as _fd, messagebox as _mb
    _root = _tk.Tk()
    _root.withdraw()
    _mb.showinfo(
        "ffmpeg が見つかりません",
        "ffmpeg.exe の場所を選択してください。\n\n"
        "ダウンロードした ffmpeg フォルダの中の\n"
        "bin\\ffmpeg.exe を選択してください。"
    )
    path = _fd.askopenfilename(
        title="ffmpeg.exe を選択",
        filetypes=[("ffmpeg", "ffmpeg.exe"), ("実行ファイル", "*.exe"), ("全て", "*")]
    )
    _root.destroy()

    if path and Path(path).exists():
        _save_ffmpeg(path)
        return path

    return None


def _save_ffmpeg(path):
    try:
        _FFMPEG_CONFIG.write_text(json.dumps({"path": str(path)}))
    except Exception:
        pass


def _setup_ffmpeg():
    """pydub に ffmpeg パスをセットする"""
    p = _find_ffmpeg()
    if p:
        AudioSegment.converter = str(p)
        # ffprobe も同じ bin フォルダにある場合はセット
        probe = Path(p).parent / (
            "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"
        )
        if probe.exists():
            AudioSegment.ffprobe = str(probe)
        return True
    return False


_FFMPEG_OK = _setup_ffmpeg()


# =====================================================
# 音楽分析エンジン
# =====================================================

SR = 44100
HOP = 512
SEMI = 2 ** (1 / 12)


class EarCopyEngine:
    """音楽分析 → 耳コピ音源生成エンジン v4.0（多声部・多楽器）"""

    def __init__(self, on_progress=None):
        self._cb = on_progress

    def _log(self, msg, pct=None):
        if self._cb:
            self._cb(msg, pct)
        else:
            print(f"[{pct or '--':>3}%] {msg}")

    # ---- 高精度CQT多声部検出 v5 ----------------------------

    def _detect_all_notes(self, y_h, sr):
        """CQT + オンセット同期 + 適応スレッショルド + 倍音除去"""
        from scipy.ndimage import median_filter, uniform_filter1d

        self._log("  CQT解析中...", 26)
        n_bins = 84
        bpo = 12
        fmin = librosa.note_to_hz('C1')
        C = np.abs(librosa.cqt(y_h, sr=sr, hop_length=HOP,
                                fmin=fmin, n_bins=n_bins, bins_per_octave=bpo))
        C_db = librosa.amplitude_to_db(C, ref=np.max)

        # 時間方向の中央値フィルタで定常ノイズ除去
        bg = median_filter(C_db, size=(1, 31))
        C_clean = C_db - bg
        C_sm = uniform_filter1d(C_clean, size=3, axis=1)

        freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bpo)
        midi_notes = np.round(librosa.hz_to_midi(freqs)).astype(int)
        times = librosa.frames_to_time(np.arange(C_sm.shape[1]),
                                        sr=sr, hop_length=HOP)

        # オンセット検出（ノートの開始タイミングを正確に）
        self._log("  オンセット検出中...", 30)
        onset_env = librosa.onset.onset_strength(y=y_h, sr=sr, hop_length=HOP)
        onsets_fr = librosa.onset.onset_detect(y=y_h, sr=sr, hop_length=HOP,
                                                onset_envelope=onset_env,
                                                backtrack=True)
        onset_set = set(onsets_fr.tolist())

        # 適応スレッショルド: 各フレームの上位N%をノートとみなす
        self._log("  ノートトラッキング中...", 33)
        active = {}
        events = []
        for fi in range(C_sm.shape[1]):
            frame = C_sm[:, fi]
            # 適応閾値: フレーム内の最大値から相対的に決定
            frame_max = np.max(frame)
            thr = max(frame_max - 22, 3.0)  # 最大値から22dB以内 + 最低3dB超え

            on_now = set()
            peaks = []
            for bi in range(1, n_bins - 1):
                if (frame[bi] > thr and
                        frame[bi] >= frame[bi-1] and frame[bi] >= frame[bi+1]):
                    peaks.append((bi, frame[bi]))

            # 倍音除去: 低い音が強い場合、その倍音を除去
            if peaks:
                peaks.sort(key=lambda x: x[1], reverse=True)
                kept = []
                used_midi = set()
                for bi, en in peaks:
                    mn = int(midi_notes[bi])
                    # 既に検出済み音の倍音(+12,+19,+24,+28,+31半音)かチェック
                    is_harmonic = False
                    for km in used_midi:
                        diff = mn - km
                        if diff in (12, 19, 24, 28, 31):
                            is_harmonic = True
                            break
                    if not is_harmonic:
                        kept.append((bi, en, mn))
                        used_midi.add(mn)
                for bi, en, mn in kept:
                    on_now.add(mn)
                    if mn not in active:
                        active[mn] = (fi, en, fi in onset_set)
                    else:
                        sf_, mx, has_onset = active[mn]
                        active[mn] = (sf_, max(mx, en), has_onset or (fi in onset_set))

            # ノートオフ
            for mn in list(active):
                if mn not in on_now:
                    sf_, mx, has_onset = active.pop(mn)
                    dur = times[min(fi, len(times)-1)] - times[sf_]
                    if dur >= 0.06:
                        vel = int(np.clip(mx * 3.5 + 30, 35, 127))
                        events.append((times[sf_], dur, mn, vel))

        for mn, (sf_, mx, has_onset) in active.items():
            dur = times[-1] - times[sf_]
            if dur >= 0.06:
                vel = int(np.clip(mx * 3.5 + 30, 35, 127))
                events.append((times[sf_], dur, mn, vel))
        return events

    def _assign_parts(self, notes):
        """検出ノートを音域別に7パートへ振り分け"""
        parts = {'melody': [], 'flute': [], 'violin': [],
                 'cello': [], 'bass': [], 'guitar': [], 'pad': []}
        for (t, dur, midi, vel) in notes:
            if midi >= 72:
                parts['melody'].append((t, dur, midi, vel))
                if vel < 85:
                    parts['flute'].append((t, dur, midi, int(vel * 0.5)))
            elif midi >= 60:
                parts['violin'].append((t, dur, midi, vel))
                parts['guitar'].append((t, dur, midi, int(vel * 0.4)))
            elif midi >= 48:
                parts['cello'].append((t, dur, midi, vel))
                parts['pad'].append((t, dur, midi, int(vel * 0.3)))
            else:
                parts['bass'].append((t, dur, midi, vel))
        return parts

    # ---- メイン処理 --------------------------------------

    def process(self, input_path: str, output_path: str):
        try:
            y, sr = self._load(input_path)
            duration = len(y) / sr
            y_h, y_p = self._hpss(y)
            tempo, beats = self._tempo(y, sr)
            self._log(f"テンポ: {tempo:.1f} BPM", 20)

            self._log("全音符を検出中（CQT多声部）...", 25)
            all_notes = self._detect_all_notes(y_h, sr)
            self._log(f"検出: {len(all_notes)} 音符", 38)

            self._log("7パートに振り分け中...", 40)
            parts = self._assign_parts(all_notes)

            self._log("ドラム解析中...", 48)
            drum_events = self._drums(y_p, sr)
            self._log(f"ドラム: {len(drum_events)} イベント", 55)

            self._log("7楽器で合成中...", 60)
            n = int((duration + 2.0) * SR)
            audio = self._synth_parts(parts, n) * 0.75
            audio += self._synth_drums(drum_events, n) * 0.55
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak * 0.9).astype(np.float32)

            self._log("MP3を保存中...", 90)
            self._save_mp3(audio, output_path)

            self._log("MIDIを保存中...", 95)
            midi_path = str(Path(output_path).with_suffix(".mid"))
            self._save_midi_v4(parts, drum_events, tempo, midi_path)

            self._log("完了！", 100)
            return True, output_path
        except Exception as e:
            import traceback
            self._log(f"エラー: {e}", -1)
            return False, traceback.format_exc()

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
        # librosa 0.10+ では tempo が配列で返る場合があるため先頭要素を取得
        tempo_val = float(np.atleast_1d(tempo)[0])
        return tempo_val, beat_times

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
            lo = fft[freqs < 120].sum() / tot
            mid = fft[(freqs >= 200) & (freqs < 1500)].sum() / tot
            hi = fft[freqs >= 3000].sum() / tot
            if lo > 0.45:
                kind = 'kick'
            elif hi > 0.4:
                kind = 'hihat'
            elif mid > 0.35 and hi > 0.15:
                kind = 'ride'
            else:
                kind = 'snare'
            events.append((ot, kind))
        return events

    # ---- 7楽器シンセサイザー --------------------------------

    def _t(self, dur):
        return np.arange(max(1, int(dur * SR))) / SR

    def _env(self, t, a=0.01, d=0.1, s=0.7, r=0.15):
        n = len(t)
        if n == 0:
            return np.array([])
        e = np.ones(n) * s
        ai = min(int(a * SR), n)
        di = min(int(d * SR), max(n - ai, 0))
        ri = min(int(r * SR), n)
        if ai > 0:
            e[:ai] = np.linspace(0, 1, ai)
        if di > 0:
            e[ai:ai+di] = np.linspace(1, s, di)
        if ri > 0:
            e[n-ri:] *= np.linspace(1, 0, ri)
        return e

    def _tone_piano(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        sig = sum((0.75**h) * np.sin(2*np.pi*hz*(h+1)*t) for h in range(8) if hz*(h+1) < SR/2)
        return sig * np.exp(-t * (1.5 + hz/600)) * self._env(t, 0.003, 0.08, 0.5, 0.12) * (vel/127) * 0.25

    def _tone_flute(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        vib = 1 + 0.003 * np.sin(2*np.pi*5.5*t)
        sig = np.sin(2*np.pi*hz*vib*t) + 0.1*np.sin(2*np.pi*hz*2*vib*t)
        sig += np.random.randn(len(t)) * 0.02 * np.exp(-t*3)
        return sig * self._env(t, 0.06, 0.1, 0.8, 0.15) * (vel/127) * 0.18

    def _tone_violin(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        vib = 1 + 0.004 * np.sin(2*np.pi*5.8*t)
        sig = sum(((-1)**h * 0.7**h) * np.sin(2*np.pi*hz*(h+1)*vib*t) for h in range(6) if hz*(h+1) < SR/2)
        return sig * self._env(t, 0.05, 0.08, 0.85, 0.12) * (vel/127) * 0.20

    def _tone_cello(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        vib = 1 + 0.003 * np.sin(2*np.pi*4.5*t)
        sig = sum((0.8**h) * np.sin(2*np.pi*hz*(h+1)*vib*t) for h in range(7) if hz*(h+1) < SR/2)
        return sig * self._env(t, 0.04, 0.1, 0.8, 0.15) * (vel/127) * 0.22

    def _tone_bass(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        sig = sum((0.85**h) * np.sin(2*np.pi*hz*(h+1)*t) for h in range(5) if hz*(h+1) < SR/2)
        return sig * np.exp(-t*1.4) * self._env(t, 0.008, 0.05, 0.75, 0.1) * (vel/127) * 0.35

    def _tone_guitar(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        sig = sum((0.7**h) * np.sin(2*np.pi*hz*(h+1)*t + h*0.3) for h in range(6) if hz*(h+1) < SR/2)
        return sig * np.exp(-t*2.5) * self._env(t, 0.002, 0.06, 0.4, 0.08) * (vel/127) * 0.20

    def _tone_pad(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        s1 = np.sin(2*np.pi*hz*t)
        s2 = np.sin(2*np.pi*hz*1.003*t)
        s3 = np.sin(2*np.pi*hz*0.997*t)
        return (s1+s2+s3)/3 * self._env(t, 0.15, 0.2, 0.6, 0.3) * (vel/127) * 0.12

    TONE_FN = {'melody': '_tone_piano', 'flute': '_tone_flute',
               'violin': '_tone_violin', 'cello': '_tone_cello',
               'bass': '_tone_bass', 'guitar': '_tone_guitar', 'pad': '_tone_pad'}
    GAIN = {'melody': 1.0, 'flute': 0.6, 'violin': 0.8, 'cello': 0.8,
            'bass': 0.9, 'guitar': 0.7, 'pad': 0.4}

    def _synth_parts(self, parts, n):
        buf = np.zeros(n)
        for name, evts in parts.items():
            fn = getattr(self, self.TONE_FN.get(name, '_tone_piano'))
            g = self.GAIN.get(name, 0.5)
            for (t, dur, midi, vel) in evts:
                s0 = int(t * SR)
                tone = fn(midi, dur, vel)
                s1 = s0 + len(tone)
                if s1 <= n:
                    buf[s0:s1] += tone * g
        return buf

    # ---- ドラム合成（4種） ---------------------------------

    def _synth_drums(self, events, n):
        buf = np.zeros(n)
        for (onset, kind) in events:
            s0 = int(onset * SR)
            snd = self._drum_sound(kind)
            s1 = s0 + len(snd)
            if s1 <= n:
                buf[s0:s1] += snd
        return buf

    def _drum_sound(self, kind):
        if kind == 'kick':
            t = self._t(0.3)
            sweep = 70 * np.exp(-t * 25)
            return (np.sin(2*np.pi*np.cumsum(sweep)/SR) * np.exp(-t*10)
                    + 0.2*np.random.randn(len(t))*np.exp(-t*30)) * 0.7
        elif kind == 'snare':
            t = self._t(0.15)
            return (0.4*np.sin(2*np.pi*200*t)*np.exp(-t*25)
                    + 0.7*np.random.randn(len(t))*np.exp(-t*20)) * 0.55
        elif kind == 'ride':
            t = self._t(0.3)
            n_ = np.random.randn(len(t))
            b, a = butter(4, min(4000/(SR/2), 0.99), btype='high')
            return filtfilt(b, a, n_) * np.exp(-t*8) * 0.25
        else:  # hihat
            t = self._t(0.04)
            n_ = np.random.randn(len(t))
            b, a = butter(4, min(6500/(SR/2), 0.99), btype='high')
            return filtfilt(b, a, n_) * np.exp(-t*100) * 0.4

    # ---- ファイル保存 ------------------------------------

    def _save_mp3(self, audio, path):
        if not _FFMPEG_OK:
            raise RuntimeError(
                "ffmpeg が見つかりませんでした。\n"
                "ツールを再起動して ffmpeg.exe の場所を選択してください。"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, audio, SR)
        seg = AudioSegment.from_wav(tmp)
        seg.export(path, format="mp3", bitrate="192k")
        os.unlink(tmp)

    def _save_midi_v4(self, parts, drums, tempo, path):
        mid = MidiFile(type=1, ticks_per_beat=480)
        tpb = 480
        us = int(60_000_000 / tempo)
        PROG = {'melody': 0, 'flute': 73, 'violin': 40, 'cello': 42,
                'bass': 32, 'guitar': 25, 'pad': 89}
        def s2t(s): return int(s * tempo / 60 * tpb)
        tt = MidiTrack()
        tt.append(MetaMessage('set_tempo', tempo=us, time=0))
        mid.tracks.append(tt)
        for ch, (name, evts) in enumerate(parts.items()):
            if ch >= 9: ch += 1
            evs = [(0, Message('program_change', channel=ch, program=PROG.get(name, 0), time=0))]
            for (t, dur, midi, vel) in evts:
                t0, t1 = s2t(t), s2t(t + dur)
                evs.append((t0, Message('note_on', channel=ch, note=int(np.clip(midi,0,127)), velocity=min(vel,127), time=0)))
                evs.append((t1, Message('note_off', channel=ch, note=int(np.clip(midi,0,127)), velocity=0, time=0)))
            trk = MidiTrack()
            evs.sort(key=lambda x: x[0])
            prev = 0
            for tick, msg in evs:
                msg.time = max(0, tick - prev)
                trk.append(msg)
                prev = tick
            mid.tracks.append(trk)
        DM = {'kick': 36, 'snare': 38, 'hihat': 42, 'ride': 51}
        devs = []
        for (t, kind) in drums:
            n = DM.get(kind, 38)
            t0 = s2t(t)
            devs.append((t0, Message('note_on', channel=9, note=n, velocity=95, time=0)))
            devs.append((t0+30, Message('note_off', channel=9, note=n, velocity=0, time=0)))
        dtrk = MidiTrack()
        devs.sort(key=lambda x: x[0])
        prev = 0
        for tick, msg in devs:
            msg.time = max(0, tick - prev)
            dtrk.append(msg)
            prev = tick
        mid.tracks.append(dtrk)
        mid.save(path)

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
