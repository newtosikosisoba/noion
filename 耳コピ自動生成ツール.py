#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
耳コピ自動生成ツール v7.0
MP3をドラッグ&ドロップするだけで原曲忠実な耳コピ音源を自動作成

v7.0 の革新:
  - Demucs (Meta): ボーカル/ドラム/ベース/その他の AI ステム分離
  - Basic Pitch (Spotify): SOTA 多声部ポリフォニック MIDI 採譜
  - FluidSynth + SoundFont: 本物のサンプル音源による再合成
  - 原曲ブレンド: 原曲ステムと合成MIDIを任意比率でミックス
  - v6.0 の加算合成はフォールバックとして残存

必要環境: Python 3.8+
初回起動時に依存パッケージを自動インストールします（AIモード時は約1-2GB）

使い方:
  GUI:  python 耳コピ自動生成ツール.py
  CLI:  python 耳コピ自動生成ツール.py input.mp3 [-o output.mp3] [--mode MODE]
        MODE: ai (Demucs+BasicPitch / デフォルト) | classic (v6.0相当) | karaoke (伴奏のみ)
"""

import os
import sys
import subprocess
import platform

# =====================================================
# 自動インストール（初回のみ）
# =====================================================

def _ensure_packages():
    """起動時に必要な軽量パッケージを確保する（AI系は遅延インストール）"""
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
                print(f"  ✗ {pkg} のインストールに失敗しました")
        print("インストール完了。起動します...\n")
        if platform.system() == "Windows":
            ret = subprocess.call([sys.executable] + sys.argv)
            sys.exit(ret)
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)


def _importable(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


# AI系の重量級パッケージは process() 実行時に遅延インストール
# 必須: torch + demucs（ステム分離の核心）
_AI_PACKAGES_REQUIRED = [
    ("torch",        "torch",       ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]),
    ("demucs",       "demucs",      ["demucs"]),
]
# 任意: basic-pitch（Python 3.13+ 非対応の場合あり、pyin でフォールバック可）
# 複数の導入方法を順に試行する（[onnx] が最も軽量、TF不要）
_AI_PACKAGES_OPTIONAL = [
    ("basic_pitch",  "basic-pitch", [
        ["basic-pitch[onnx]"],                                       # 軽量: onnxruntime
        ["basic-pitch[tflite]"],                                     # 軽量: TFLite
        ["basic-pitch[coreml]"],                                     # macOS専用
        ["basic-pitch"],                                             # フル: TensorFlow
        ["basic-pitch", "--no-deps"],                                # 依存衝突回避
    ]),
]


def _ensure_ai_packages(log=print):
    """AIモード起動時に重量級パッケージを確保する（初回のみ大きなダウンロード）"""
    # 必須パッケージ
    missing = [(imp, pkg, args) for imp, pkg, args in _AI_PACKAGES_REQUIRED
               if not _importable(imp)]
    if missing:
        log(f"AI必須パッケージを導入中 (初回のみ、約1-2GB): {', '.join(p for _, p, _ in missing)}")
        for imp, pkg, args in missing:
            log(f"  インストール中: {pkg} ...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q"] + args,
                    stderr=subprocess.STDOUT
                )
                log(f"  ✓ {pkg}")
            except subprocess.CalledProcessError as e:
                log(f"  ✗ {pkg} のインストールに失敗: {e}")
                return False

    # 任意パッケージ（失敗しても続行、複数の導入方法を順に試行）
    for imp, pkg, install_variants in _AI_PACKAGES_OPTIONAL:
        if _importable(imp):
            continue
        log(f"  オプション: {pkg} を導入中 ({len(install_variants)}方法を順に試行)...")
        success = False
        last_error = ""
        for variant_idx, args in enumerate(install_variants, 1):
            variant_name = " ".join(args)
            log(f"    [{variant_idx}/{len(install_variants)}] pip install {variant_name}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + args,
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    log(f"  ✓ {pkg} ({variant_name})")
                    success = True
                    break
                # stderr の末尾を最大 300 文字表示してエラー原因を見せる
                err_tail = (result.stderr or result.stdout or "").strip().splitlines()
                tail = "\n      ".join(err_tail[-4:]) if err_tail else "unknown"
                last_error = tail
                log(f"    ✗ 失敗:\n      {tail}")
            except subprocess.TimeoutExpired:
                last_error = "timeout (5min)"
                log(f"    ✗ タイムアウト")
            except Exception as e:
                last_error = str(e)
                log(f"    ✗ {e}")
        if not success:
            log(f"  ⚠ {pkg} は利用不可（pyin+CQTで代替します）")
            log(f"    最終エラー: {last_error[:200]}")
    return True


try:
    _ensure_packages()
except Exception as e:
    print(f"パッケージセットアップ中にエラー: {e}")
    input("Enterキーで終了...")
    sys.exit(1)

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
import threading
import tempfile
import shutil
import json
from pathlib import Path

# tkinter は GUI モード時のみインポート（CLIモードでは不要）
tk = None
ttk = None
messagebox = None
filedialog = None
HAS_DND = False
TkinterDnD = None
DND_FILES = None


def _init_gui():
    """GUIモード起動時にtkinter関連をインポートする"""
    global tk, ttk, messagebox, filedialog, HAS_DND, TkinterDnD, DND_FILES
    import tkinter as _tk
    from tkinter import ttk as _ttk, messagebox as _mb, filedialog as _fd
    tk = _tk
    ttk = _ttk
    messagebox = _mb
    filedialog = _fd
    try:
        from tkinterdnd2 import DND_FILES as _dnd, TkinterDnD as _TkDnD
        HAS_DND = True
        TkinterDnD = _TkDnD
        DND_FILES = _dnd
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
    try:
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
    except ImportError:
        print("警告: ffmpeg が見つかりません。PATHにffmpegを追加してください。")

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
# SoundFont / FluidSynth 自動セットアップ
# =====================================================

_SF2_CONFIG = Path(__file__).parent / ".sf2_path.json"
_ASSETS_DIR = Path(__file__).parent / "_assets"

# 無料で商用可な汎用GM SoundFont（失敗時の候補を複数用意）
_SF2_DOWNLOADS = [
    # (表示名, URL, 想定サイズMB, 拡張子)
    ("FluidR3_GM (SourceForge)",
     "https://sourceforge.net/projects/pianobooster/files/pianobooster/1.0.0/FluidR3_GM.sf2/download",
     140, "sf2"),
    ("FluidR3_GM (GitHub)",
     "https://github.com/urish/cinto/raw/master/media/FluidR3%20GM.sf2",
     140, "sf2"),
    ("FluidR3_GM (Musical Artifacts)",
     "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2",
     140, "sf2"),
]


def _find_sf2():
    """SoundFont (.sf2/.sf3) のパスを解決する"""
    # 1. 前回保存したパス
    if _SF2_CONFIG.exists():
        try:
            saved = json.loads(_SF2_CONFIG.read_text())
            p = saved.get("path", "")
            if p and Path(p).exists():
                return p
        except Exception:
            pass

    # 2. アセットディレクトリを検索
    if _ASSETS_DIR.exists():
        for ext in ("*.sf2", "*.sf3"):
            for p in _ASSETS_DIR.glob(ext):
                return str(p)

    # 3. よくあるシステム設置場所
    candidates = []
    if platform.system() == "Windows":
        candidates += [
            r"C:\ProgramData\soundfonts\default.sf2",
            r"C:\Program Files\MuseScore 4\sound\MS Basic.sf3",
            r"C:\Program Files\MuseScore 3\sound\MuseScore_General.sf3",
        ]
    elif platform.system() == "Darwin":
        candidates += [
            "/Applications/MuseScore 4.app/Contents/Resources/sound/MS Basic.sf3",
            "/Library/Audio/Sounds/Banks/default.sf2",
        ]
    else:
        candidates += [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2",
            "/usr/share/soundfonts/default.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
        ]
    for c in candidates:
        if Path(c).exists():
            _save_sf2(c)
            return c
    return None


def _save_sf2(path):
    try:
        _SF2_CONFIG.write_text(json.dumps({"path": str(path)}))
    except Exception:
        pass


def _download_sf2(log=print):
    """SoundFont が無ければ自動ダウンロードする（複数URLを順に試行）"""
    _ASSETS_DIR.mkdir(exist_ok=True)
    import urllib.request
    import zipfile

    for name, url, size, ext in _SF2_DOWNLOADS:
        log(f"  SoundFont をダウンロード中 ({name}, 約{size}MB)...")
        try:
            fname = url.rsplit("/", 1)[-1].replace("%20", "_")
            if not fname.lower().endswith((".sf2", ".sf3", ".zip")):
                fname = f"{name.replace(' ', '_')}.{ext}"
            dest = _ASSETS_DIR / fname

            # User-Agent を付けないと弾くサーバーがある
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (EarCopyTool)"
            })
            with urllib.request.urlopen(req, timeout=60) as r:
                ct = r.headers.get("Content-Type", "").lower()
                # HTMLが返ってきた場合は失敗扱い（リダイレクト先の案内ページなど）
                if "text/html" in ct:
                    raise RuntimeError(f"HTML応答を受信 (Content-Type: {ct})")
                data = r.read()
            if len(data) < 100_000:  # 100KB未満は怪しい
                raise RuntimeError(f"ファイルサイズが小さすぎます ({len(data)} bytes)")
            dest.write_bytes(data)

            # ZIP判定はヘッダで確実に
            is_zip = dest.suffix.lower() == ".zip" or data[:4] == b"PK\x03\x04"
            if is_zip:
                with zipfile.ZipFile(dest) as zf:
                    for member in zf.namelist():
                        if member.lower().endswith((".sf2", ".sf3")):
                            zf.extract(member, _ASSETS_DIR)
                            extracted = _ASSETS_DIR / member
                            _save_sf2(str(extracted))
                            log(f"  ✓ {extracted.name}")
                            return str(extracted)
                dest.unlink(missing_ok=True)
                log(f"  ✗ {name}: ZIPにSF2が含まれず")
                continue

            # SF2ヘッダ検証 (RIFF...sfbk)
            if len(data) > 16 and (data[:4] == b"RIFF" and b"sfbk" in data[:64]):
                _save_sf2(str(dest))
                log(f"  ✓ {dest.name}")
                return str(dest)
            dest.unlink(missing_ok=True)
            log(f"  ✗ {name}: SF2フォーマット不正")
        except Exception as e:
            log(f"  ✗ {name} 取得失敗: {e}")
            continue
    return None


def _prompt_sf2_manually(log=print):
    """自動DLが全滅した場合、ユーザーに手動選択を促す"""
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd, messagebox as _mb
        _root = _tk.Tk()
        _root.withdraw()
        yes = _mb.askyesno(
            "SoundFontの自動取得に失敗",
            "SoundFont (.sf2/.sf3) ファイルを手動で指定しますか？\n\n"
            "【いいえ】を選ぶと v6.0 加算合成で音源を生成します。\n\n"
            "【はい】を選ぶとファイル選択ダイアログが開きます。\n"
            "お持ちでなければ、以下から無料ダウンロードできます:\n"
            "  • https://member.keymusician.com/Member/FluidR3_GM/\n"
            "  • https://schristiancollins.com/generaluser.php"
        )
        if not yes:
            _root.destroy()
            return None
        path = _fd.askopenfilename(
            title="SoundFont (.sf2 / .sf3) を選択",
            filetypes=[("SoundFont", "*.sf2 *.sf3"), ("全て", "*")]
        )
        _root.destroy()
        if path and Path(path).exists():
            _save_sf2(path)
            log(f"  ✓ 手動指定: {Path(path).name}")
            return path
    except Exception:
        pass
    return None


def _setup_fluidsynth():
    """pyfluidsynth の利用可否を判定する"""
    if not _importable("fluidsynth"):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pyfluidsynth", "-q"],
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return False
    try:
        import fluidsynth  # noqa
        return True
    except Exception:
        return False


# =====================================================
# 音楽分析エンジン
# =====================================================

SR = 44100
HOP = 512
SEMI = 2 ** (1 / 12)


class EarCopyEngine:
    """v7.0 - Demucs + Basic Pitch + FluidSynth を主軸とした原曲忠実エンジン

    モード:
      - "ai"      : Demucs でステム分離 → Basic Pitch で多声部採譜 → FluidSynth 合成
                    （未セットアップ時は v6 加算合成にフォールバック）
      - "karaoke" : Demucs でボーカル除去のみ（伴奏はそのまま出力、最も原曲に近い）
      - "classic" : v6.0 同等の CQT+pyin 解析＆加算合成（依存最小）
      - "blend"   : 原曲ステム + 合成MIDI を指定比率でブレンド

    blend_ratio: 0.0 (合成のみ) 〜 1.0 (原曲のみ)
    """

    MIDI_MAP = {
        'piano': (0, 0), 'e_piano': (1, 4), 'glockenspiel': (2, 9),
        'organ': (3, 19), 'guitar_nylon': (4, 24), 'guitar_clean': (5, 27),
        'bass': (6, 33), 'violin': (7, 40), 'viola': (8, 41),
        'cello': (10, 42), 'strings': (11, 48), 'choir': (12, 52),
        'trumpet': (13, 56), 'flute': (14, 73), 'pad': (15, 89),
    }
    GAIN = {
        'piano': 1.0, 'e_piano': 0.7, 'glockenspiel': 0.5,
        'organ': 0.5, 'guitar_nylon': 0.7, 'guitar_clean': 0.6,
        'bass': 0.9, 'violin': 0.8, 'viola': 0.7, 'cello': 0.8,
        'strings': 0.5, 'choir': 0.4, 'trumpet': 0.6,
        'flute': 0.6, 'pad': 0.4,
    }
    CHORD_IV = {
        'maj': [0,4,7], 'min': [0,3,7], 'dom7': [0,4,7,10],
        'min7': [0,3,7,10], 'maj7': [0,4,7,11], 'dim': [0,3,6],
    }
    NOTE_MIDI = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,
                 'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}

    def __init__(self, on_progress=None, mode="ai", blend_ratio=0.0):
        self._cb = on_progress
        self.mode = mode  # "ai" | "karaoke" | "classic" | "blend"
        self.blend_ratio = max(0.0, min(1.0, blend_ratio))

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

    # ---- pyin メロディ検出 --------------------------------

    def _detect_melody(self, y_h, sr):
        self._log("  メロディ解析中 (pyin)...", 33)
        f0, voiced, _ = librosa.pyin(
            y_h, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=HOP)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
        return self._f0_to_notes(f0, voiced, times, 8)

    # ---- pyin ベース検出 ----------------------------------

    def _detect_bass(self, y_h, sr):
        self._log("  ベースライン解析中...", 37)
        nyq = sr / 2
        b, a = butter(4, min(250 / nyq, 0.99), btype="low")
        yb = filtfilt(b, a, y_h)
        f0, voiced, _ = librosa.pyin(
            yb, fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C3"), sr=sr, hop_length=HOP * 2)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP * 2)
        return self._f0_to_notes(f0, voiced, times, 4)

    def _f0_to_notes(self, f0, voiced, times, max_gap_hz):
        events = []
        i = 0
        while i < len(f0):
            if voiced[i] and f0[i] is not None and not np.isnan(f0[i]):
                start_t = times[i]
                base_hz = f0[i]
                hz_list = [base_hz]
                j = i + 1
                while (j < len(f0) and voiced[j] and
                       f0[j] is not None and not np.isnan(f0[j]) and
                       abs(f0[j] - base_hz) < max_gap_hz):
                    hz_list.append(f0[j])
                    j += 1
                dur = times[min(j, len(times)-1)] - start_t
                if dur >= 0.06:
                    midi = int(np.clip(np.round(
                        librosa.hz_to_midi(np.mean(hz_list))), 0, 127))
                    vel = min(int(80 + dur * 10), 120)
                    events.append((start_t, min(dur, 4.0), midi, vel))
                i = j
            else:
                i += 1
        return events

    # ---- コード検出 ----------------------------------------

    def _detect_chords(self, y_h, sr, beat_times):
        self._log("  コード解析中...", 42)
        if len(beat_times) == 0:
            return []
        chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP)
        NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        TEMPLATES = {
            'maj': [1,0,0,0,1,0,0,1,0,0,0,0],
            'min': [1,0,0,1,0,0,0,1,0,0,0,0],
            'dom7':[1,0,0,0,1,0,0,1,0,0,1,0],
            'min7':[1,0,0,1,0,0,0,1,0,0,1,0],
            'dim': [1,0,0,1,0,0,1,0,0,0,0,0],
        }
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
        events = []
        for idx, bf in enumerate(beat_frames):
            end = (beat_frames[idx+1] if idx+1 < len(beat_frames)
                   else min(bf+8, chroma.shape[1]-1))
            if bf >= chroma.shape[1]:
                break
            cv = np.mean(chroma[:, bf:end+1], axis=1)
            cv = cv / (cv.max() + 1e-9)
            best, chord = -1, ("C", "maj")
            for root in range(12):
                for q, tmpl in TEMPLATES.items():
                    s = float(np.dot(cv, np.roll(tmpl, root)))
                    if s > best:
                        best, chord = s, (NOTES[root], q)
            next_t = (beat_times[idx+1] if idx+1 < len(beat_times)
                      else beat_times[idx] + 0.5)
            dur = max(next_t - beat_times[idx] - 0.02, 0.05)
            events.append((beat_times[idx], chord[0], chord[1], dur))
        return events

    def _chord_to_midis(self, root, quality, octave=4):
        root_midi = self.NOTE_MIDI.get(root, 0) + 12 * (octave + 1)
        return [root_midi + iv for iv in self.CHORD_IV.get(quality, [0,4,7])]

    # ---- 15楽器スマート振り分け ----------------------------

    def _smart_assign(self, cqt_notes, melody, bass, chords, beat_times):
        self._log("15パートに振り分け中...", 55)
        parts = {name: [] for name in self.MIDI_MAP}
        mel_set = set()
        bass_set = set()

        # 1. メロディ → ピアノ + ダブリング
        for (t, dur, midi, vel) in melody:
            parts['piano'].append((t, dur, midi, vel))
            mel_set.add((round(t, 2), midi))
            if midi >= 80:
                parts['flute'].append((t, dur, midi, int(vel * 0.4)))
            elif midi >= 68:
                parts['violin'].append((t, dur, midi, int(vel * 0.4)))
            else:
                parts['e_piano'].append((t, dur, midi, int(vel * 0.35)))

        # 2. ベース → ベース + チェロ
        for (t, dur, midi, vel) in bass:
            parts['bass'].append((t, dur, midi, int(vel * 0.95)))
            bass_set.add((round(t, 2), midi))
            if midi + 12 < 60:
                parts['cello'].append((t, dur, midi + 12, int(vel * 0.45)))

        # 3. コード → ギター + ストリングス + パッド + オルガン
        prev_chord = None
        for (t, root, quality, dur) in chords:
            c4 = self._chord_to_midis(root, quality, 4)
            c3 = self._chord_to_midis(root, quality, 3)
            for cm in c4:
                parts['guitar_nylon'].append((t, dur * 0.9, cm, 60))
                parts['strings'].append((t, dur, cm, 42))
            for cm in c3:
                parts['pad'].append((t, dur, cm, 32))
            cur = (root, quality)
            if cur != prev_chord:
                for cm in c3:
                    parts['organ'].append((t, dur, cm, 28))
            prev_chord = cur

        # 4. CQT残り → 音域で分配
        for (t, dur, midi, vel) in cqt_notes:
            key = (round(t, 2), midi)
            if key in mel_set or key in bass_set:
                continue
            skip = False
            for mt, mm in mel_set:
                if abs(round(t, 2) - mt) < 0.05 and abs(midi - mm) <= 2:
                    skip = True
                    break
            if skip:
                continue
            if midi >= 84:
                parts['glockenspiel'].append((t, dur, midi, int(vel * 0.35)))
            elif midi >= 72:
                if dur >= 0.4:
                    parts['violin'].append((t, dur, midi, int(vel * 0.5)))
                    parts['choir'].append((t, dur, midi, int(vel * 0.2)))
                else:
                    parts['e_piano'].append((t, dur, midi, int(vel * 0.5)))
            elif midi >= 60:
                if dur >= 0.4:
                    parts['viola'].append((t, dur, midi, int(vel * 0.5)))
                else:
                    parts['guitar_clean'].append((t, dur, midi, int(vel * 0.45)))
            elif midi >= 48:
                if dur >= 0.4:
                    parts['cello'].append((t, dur, midi, int(vel * 0.55)))
                else:
                    parts['guitar_clean'].append((t, dur, midi, int(vel * 0.4)))

        # 5. 強拍トランペットアクセント
        if beat_times is not None and len(melody) > 0:
            for i, bt in enumerate(beat_times):
                if i % 4 != 0:
                    continue
                for (mt, md, mm, mv) in melody:
                    if abs(mt - bt) < 0.08 and mm >= 60:
                        parts['trumpet'].append((mt, min(md, 0.4), mm, int(mv * 0.3)))
                        break

        return parts

    # ---- メイン処理ディスパッチャー ---------------------

    def process(self, input_path: str, output_path: str):
        """モードに応じて処理を分岐する"""
        try:
            if self.mode == "classic":
                return self._process_classic(input_path, output_path)
            if self.mode == "karaoke":
                return self._process_karaoke(input_path, output_path)
            # "ai" または "blend" はAIパイプライン
            return self._process_ai(input_path, output_path)
        except Exception as e:
            import traceback
            self._log(f"エラー: {e}", -1)
            return False, traceback.format_exc()

    # ---- Classic (v6.0 互換) ----------------------------

    def _process_classic(self, input_path: str, output_path: str):
        try:
            y, sr = self._load(input_path)
            duration = len(y) / sr
            y_h, y_p = self._hpss(y)
            tempo, beats = self._tempo(y, sr)
            self._log(f"テンポ: {tempo:.1f} BPM", 20)

            # 4層検出
            self._log("CQT多声部解析中...", 24)
            cqt_notes = self._detect_all_notes(y_h, sr)
            self._log(f"  CQT: {len(cqt_notes)} 音符", 32)

            melody = self._detect_melody(y_h, sr)
            self._log(f"  メロディ: {len(melody)} 音符", 35)

            bass = self._detect_bass(y_h, sr)
            self._log(f"  ベース: {len(bass)} 音符", 40)

            chords = self._detect_chords(y_h, sr, beats)
            self._log(f"  コード: {len(chords)} 進行", 45)

            self._log("ドラム解析中...", 48)
            drum_events = self._drums(y_p, sr)
            self._log(f"  ドラム: {len(drum_events)} イベント", 52)

            parts = self._smart_assign(cqt_notes, melody, bass, chords, beats)
            total = sum(len(v) for v in parts.values())
            self._log(f"  全パート合計: {total} ノート（15楽器）", 58)

            self._log("15楽器+ドラムで合成中...", 60)
            n = int((duration + 2.0) * SR)
            audio = self._synth_parts(parts, n) * 0.75
            audio += self._synth_drums(drum_events, n) * 0.50
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak * 0.92).astype(np.float32)

            self._log("MP3を保存中...", 90)
            self._save_mp3(audio, output_path)

            self._log("MIDIを保存中...", 95)
            midi_path = str(Path(output_path).with_suffix(".mid"))
            self._save_midi(parts, drum_events, tempo, midi_path)

            self._log("完了！", 100)
            return True, output_path
        except Exception as e:
            import traceback
            self._log(f"エラー: {e}", -1)
            return False, traceback.format_exc()

    # ============================================================
    # v7.0: AI パイプライン (Demucs + Basic Pitch + FluidSynth)
    # ============================================================

    def _process_ai(self, input_path: str, output_path: str):
        """Demucs + Basic Pitch の主パイプライン"""
        self._log("AIモデル準備中...", 2)
        if not _ensure_ai_packages(lambda m: self._log(m, 3)):
            self._log("AIパッケージ取得に失敗、Classic モードへフォールバック", 5)
            return self._process_classic(input_path, output_path)

        # 1. 原音ロード
        y, sr = self._load(input_path)
        duration = len(y) / sr

        # 2. Demucs でステム分離
        self._log("Demucs でステム分離中 (初回はモデル~80MBをダウンロード)...", 8)
        stems = self._demucs_separate(input_path)
        if stems is None:
            self._log("Demucs 失敗、Classic モードへフォールバック", 12)
            return self._process_classic(input_path, output_path)

        # 3. 各ステムから採譜
        tempo, beats = self._tempo(y, sr)
        self._log(f"テンポ: {tempo:.1f} BPM", 28)

        self._log("ボーカル/その他を多声部採譜中 (Basic Pitch)...", 32)
        vocal_notes = self._basic_pitch_notes(stems["vocals"], sr, is_vocal=True)
        self._log(f"  ボーカル: {len(vocal_notes)} 音符", 45)

        other_notes = self._basic_pitch_notes(stems["other"], sr, is_vocal=False)
        self._log(f"  その他: {len(other_notes)} 音符", 58)

        self._log("ベースライン採譜中 (pyin)...", 62)
        bass_notes = self._pyin_bass_notes(stems["bass"], sr)
        self._log(f"  ベース: {len(bass_notes)} 音符", 68)

        self._log("ドラム採譜中...", 70)
        drum_events = self._drums_from_stem(stems["drums"], sr)
        self._log(f"  ドラム: {len(drum_events)} イベント", 74)

        # 4. AI検出結果を15楽器パートに割り当て
        parts = self._ai_assign_parts(vocal_notes, other_notes, bass_notes)

        # 5. MIDI 保存
        self._log("MIDIを保存中...", 78)
        midi_path = str(Path(output_path).with_suffix(".mid"))
        self._save_midi(parts, drum_events, tempo, midi_path)

        # 6. 音声合成: FluidSynth 優先、失敗時は加算合成
        self._log("音声合成中...", 82)
        audio = self._synthesize_audio(midi_path, parts, drum_events, duration)

        # 7. ブレンドモード / karaoke 合成
        if self.mode == "blend" and self.blend_ratio > 0:
            self._log(f"原曲と合成をブレンド中 (原曲 {self.blend_ratio*100:.0f}%)...", 88)
            mix = self._mix_stems(stems, include_vocals=False)
            audio = self._blend(audio, mix, sr, self.blend_ratio)

        # 8. マスタリング
        audio = self._master(audio)

        # 9. 保存
        self._log("MP3を保存中...", 94)
        self._save_mp3(audio, output_path)

        self._log("完了！", 100)
        return True, output_path

    def _process_karaoke(self, input_path: str, output_path: str):
        """ボーカル除去のみ（原曲クオリティそのまま）"""
        self._log("AIモデル準備中...", 2)
        if not _ensure_ai_packages(lambda m: self._log(m, 3)):
            return False, "AIパッケージが必要です"

        y, sr = self._load(input_path)
        duration = len(y) / sr

        self._log("Demucs でボーカル分離中...", 15)
        stems = self._demucs_separate(input_path)
        if stems is None:
            return False, "Demucs でのステム分離に失敗しました"

        self._log("伴奏をミックス中...", 75)
        inst = self._mix_stems(stems, include_vocals=False)
        inst = self._master(inst)

        # MIDI も参考用に出力
        tempo, _ = self._tempo(y, sr)
        self._log("参考 MIDI を作成中...", 85)
        bass_notes = self._pyin_bass_notes(stems["bass"], sr)
        other_notes = self._basic_pitch_notes(stems["other"], sr, is_vocal=False)
        vocal_notes = self._basic_pitch_notes(stems["vocals"], sr, is_vocal=True)
        drum_events = self._drums_from_stem(stems["drums"], sr)
        parts = self._ai_assign_parts(vocal_notes, other_notes, bass_notes)
        midi_path = str(Path(output_path).with_suffix(".mid"))
        self._save_midi(parts, drum_events, tempo, midi_path)

        self._log("MP3を保存中...", 94)
        self._save_mp3(inst, output_path)
        self._log("完了！", 100)
        return True, output_path

    # ---- Demucs ステム分離 --------------------------------

    def _demucs_separate(self, input_path):
        """Demucs (htdemucs) でステム分離し、dict{name: ndarray} を返す"""
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            from demucs.audio import AudioFile, convert_audio
        except Exception as e:
            self._log(f"Demucs インポート失敗: {e}", -1)
            return None

        try:
            model = get_model("htdemucs")
            model.cpu()
            model.eval()

            wav = AudioFile(input_path).read(
                streams=0, samplerate=model.samplerate, channels=model.audio_channels
            )
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / (ref.std() + 1e-8)

            with torch.no_grad():
                sources = apply_model(
                    model, wav[None], device="cpu",
                    split=True, overlap=0.15, progress=False
                )[0]
            sources = sources * ref.std() + ref.mean()

            # htdemucs: sources は [drums, bass, other, vocals]
            names = model.sources
            out = {}
            for name, src in zip(names, sources):
                # モノラル化して SR にリサンプル
                audio = src.mean(0).cpu().numpy()
                if model.samplerate != SR:
                    audio = librosa.resample(audio, orig_sr=model.samplerate, target_sr=SR)
                out[name] = audio.astype(np.float32)
            return out
        except Exception as e:
            import traceback
            self._log(f"Demucs 実行エラー: {e}", -1)
            traceback.print_exc()
            return None

    def _mix_stems(self, stems, include_vocals=True, gains=None):
        """ステムをモノミックスする"""
        if gains is None:
            gains = {"drums": 1.0, "bass": 1.0, "other": 1.0, "vocals": 1.0}
        if not include_vocals:
            gains = {**gains, "vocals": 0.0}
        max_len = max(len(s) for s in stems.values())
        mix = np.zeros(max_len, dtype=np.float32)
        for name, s in stems.items():
            g = gains.get(name, 1.0)
            if g == 0 or len(s) == 0:
                continue
            mix[:len(s)] += s * g
        return mix

    # ---- Basic Pitch 多声部採譜 ---------------------------

    def _basic_pitch_notes(self, audio, sr, is_vocal=False):
        """Basic Pitch で多声部採譜し [(t, dur, midi, vel), ...] を返す"""
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except Exception as e:
            self._log(f"Basic Pitch インポート失敗、fallback: {e}", -1)
            return self._fallback_transcribe(audio, sr, is_vocal)

        # Basic Pitch は 22050Hz を期待
        target_sr = 22050
        if sr != target_sr:
            audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_22k = audio

        # Basic Pitch は ndarray を直接受け付けない版もあるため wav 一時ファイル経由
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, audio_22k, target_sr)
            try:
                _, _, note_events = predict(
                    tmp,
                    model_or_model_path=str(ICASSP_2022_MODEL_PATH),
                    onset_threshold=0.5 if is_vocal else 0.4,
                    frame_threshold=0.3,
                    minimum_note_length=80 if is_vocal else 60,
                    minimum_frequency=65.0 if is_vocal else 32.0,
                    maximum_frequency=2000.0 if is_vocal else 4000.0,
                    melodia_trick=is_vocal,
                    midi_tempo=120,
                )
            except TypeError:
                # 古いAPI用の引数縮小
                _, _, note_events = predict(tmp)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

        out = []
        for ev in note_events:
            # note_events は (start, end, pitch, amplitude, pitch_bends)
            start = float(ev[0])
            end = float(ev[1])
            midi = int(ev[2])
            amp = float(ev[3]) if len(ev) > 3 else 0.8
            dur = max(end - start, 0.05)
            vel = int(np.clip(40 + amp * 80, 30, 127))
            out.append((start, dur, midi, vel))
        return out

    def _fallback_transcribe(self, audio, sr, is_vocal):
        """Basic Pitch が使えない場合: CQT多声部 + pyin を分離済みステムに適用

        Demucsで分離された各ステム（ボーカルだけ/その他だけ）に対してCQTを
        かけるため、v6.0の混合音全体解析よりも格段に精度が高い。
        """
        notes = []
        # pyin で主旋律（単音）を検出
        fmin = librosa.note_to_hz("C3" if is_vocal else "C2")
        fmax = librosa.note_to_hz("C6" if is_vocal else "C7")
        try:
            f0, voiced, _ = librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=HOP)
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
            notes = self._f0_to_notes(f0, voiced, times, 8)
        except Exception:
            pass

        # ボーカルでなければ CQT で多声部も追加検出
        if not is_vocal:
            try:
                y_h, _ = librosa.effects.hpss(audio, margin=2.0)
                cqt_notes = self._detect_all_notes(y_h, sr)
                # pyin と重複しないノートだけ追加
                pyin_set = {(round(t, 2), m) for t, _, m, _ in notes}
                for (t, dur, midi, vel) in cqt_notes:
                    key = (round(t, 2), midi)
                    if key not in pyin_set:
                        notes.append((t, dur, midi, vel))
            except Exception:
                pass

        return notes

    # ---- pyin ベース採譜 (Demucs ベースステム用) ----------

    def _pyin_bass_notes(self, audio, sr):
        """Demucs で分離されたベース音声を pyin で採譜"""
        if np.max(np.abs(audio)) < 1e-4:
            return []
        f0, voiced, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C3"), sr=sr, hop_length=HOP * 2
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP * 2)
        return self._f0_to_notes(f0, voiced, times, 4)

    # ---- ドラム (Demucs ドラムステム直接解析) --------------

    def _drums_from_stem(self, audio, sr):
        """Demucs で分離されたドラムステムをオンセット検出＋スペクトル分類"""
        if np.max(np.abs(audio)) < 1e-4:
            return []
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, hop_length=HOP, backtrack=True, delta=0.06
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)
        events = []
        for ot, of_ in zip(onset_times, onset_frames):
            s = max(0, of_ - 2) * HOP
            e = min(of_ + 8, len(audio) // HOP) * HOP
            seg = audio[s:e]
            if len(seg) < 8:
                continue
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1 / sr)
            tot = fft.sum() + 1e-9
            lo = fft[freqs < 120].sum() / tot
            mid = fft[(freqs >= 200) & (freqs < 1500)].sum() / tot
            hi = fft[freqs >= 3000].sum() / tot
            if lo > 0.45:
                kind = "kick"
            elif hi > 0.45:
                kind = "hihat"
            elif mid > 0.3 and hi > 0.15:
                kind = "ride"
            else:
                kind = "snare"
            events.append((ot, kind))
        return events

    # ---- AI 検出結果 → 15楽器割り当て ----------------------

    def _ai_assign_parts(self, vocal_notes, other_notes, bass_notes):
        """AI採譜結果を v6 互換の15パート形式に割り当てる"""
        parts = {name: [] for name in self.MIDI_MAP}

        # ボーカル系は主旋律 → ピアノ(主) + フルート/バイオリン(エコー)
        for (t, dur, midi, vel) in vocal_notes:
            parts['piano'].append((t, dur, midi, vel))
            if midi >= 78:
                parts['flute'].append((t, dur, midi, int(vel * 0.45)))
            elif midi >= 64:
                parts['violin'].append((t, dur, midi, int(vel * 0.5)))
            else:
                parts['choir'].append((t, dur, midi, int(vel * 0.3)))

        # その他(伴奏) → 音域で楽器分け
        for (t, dur, midi, vel) in other_notes:
            if midi >= 84:
                parts['glockenspiel'].append((t, dur, midi, int(vel * 0.5)))
            elif midi >= 72:
                parts['e_piano'].append((t, dur, midi, int(vel * 0.6)))
                if dur >= 0.35:
                    parts['strings'].append((t, dur, midi, int(vel * 0.35)))
            elif midi >= 60:
                parts['guitar_clean'].append((t, dur, midi, int(vel * 0.7)))
                if dur >= 0.4:
                    parts['strings'].append((t, dur, midi, int(vel * 0.3)))
            elif midi >= 48:
                parts['guitar_nylon'].append((t, dur, midi, int(vel * 0.6)))
                if dur >= 0.5:
                    parts['pad'].append((t, dur, midi, int(vel * 0.3)))
            else:
                parts['cello'].append((t, dur, midi, int(vel * 0.7)))

        # ベース
        for (t, dur, midi, vel) in bass_notes:
            parts['bass'].append((t, dur, midi, vel))
            if midi + 12 < 55:
                parts['cello'].append((t, dur, midi + 12, int(vel * 0.4)))

        return parts

    # ---- 合成 (FluidSynth 優先) ---------------------------

    def _synthesize_audio(self, midi_path, parts, drum_events, duration):
        """FluidSynth で合成、失敗時は v6 加算合成にフォールバック"""
        audio = self._synthesize_fluidsynth(midi_path, duration)
        if audio is not None:
            return audio * 0.85
        self._log("  FluidSynth 未使用、v6加算合成を使用", 84)
        n = int((duration + 2.0) * SR)
        audio = self._synth_parts(parts, n) * 0.75
        audio += self._synth_drums(drum_events, n) * 0.50
        return audio

    def _synthesize_fluidsynth(self, midi_path, duration):
        """FluidSynth + SoundFont で MIDI を合成する"""
        sf2 = _find_sf2()
        if sf2 is None:
            self._log("  SoundFont が見つからないため自動取得を試みます...", 82)
            sf2 = _download_sf2(lambda m: self._log(m, 83))
        if sf2 is None:
            self._log("  自動取得失敗、手動選択ダイアログを表示します...", 84)
            sf2 = _prompt_sf2_manually(lambda m: self._log(m, 84))
        if sf2 is None:
            self._log("  SoundFontなし、v6加算合成を使用します", 84)
            return None
        if not _setup_fluidsynth():
            self._log("  FluidSynthライブラリ未検出、v6加算合成を使用します", 84)
            return None
        try:
            import fluidsynth
        except Exception:
            return None

        try:
            fs = fluidsynth.Synth(samplerate=float(SR), gain=0.6)
            sfid = fs.sfload(sf2)
            if sfid == -1:
                return None

            # MIDI をパースして直接 fluidsynth に送る
            mid = MidiFile(midi_path)

            # 各チャンネルの program を pre-assign
            for ch, prog in [(info[0], info[1]) for info in self.MIDI_MAP.values()]:
                fs.program_select(ch, sfid, 0, prog)
            # ドラム (ch9) は bank 128 (GM drum kit)
            try:
                fs.program_select(9, sfid, 128, 0)
            except Exception:
                pass

            # リアルタイム代わりに非リアルタイム書き出し
            # pyfluidsynth は get_samples(n) で合成できる
            total_seconds = mid.length + 2.0
            total_frames = int(total_seconds * SR)
            buf = np.zeros(total_frames * 2, dtype=np.int16)  # ステレオ

            # イベントをタイムスタンプ付きでソートして逐次送信＆合成
            events = []  # (time_sec, msg)
            cur_time = {i: 0.0 for i in range(len(mid.tracks))}
            tempo_us = 500_000  # デフォルト 120BPM
            tpb = mid.ticks_per_beat
            # 全トラック統合
            merged = []
            for ti, tr in enumerate(mid.tracks):
                abs_ticks = 0
                for msg in tr:
                    abs_ticks += msg.time
                    merged.append((abs_ticks, msg))
            merged.sort(key=lambda x: x[0])

            def ticks_to_sec(ticks, tempo):
                return ticks * (tempo / 1_000_000.0) / tpb

            cur_ticks = 0
            cur_sec = 0.0
            cursor_frame = 0
            for abs_ticks, msg in merged:
                dt_ticks = abs_ticks - cur_ticks
                dt_sec = ticks_to_sec(dt_ticks, tempo_us)
                cur_ticks = abs_ticks
                # dt_sec 分 fluidsynth から samples を取得
                n_frames = int(dt_sec * SR)
                if n_frames > 0:
                    samples = fs.get_samples(n_frames)
                    end = cursor_frame + n_frames * 2
                    if end > len(buf):
                        end = len(buf)
                        samples = samples[: end - cursor_frame]
                    buf[cursor_frame:end] = samples[: end - cursor_frame]
                    cursor_frame = end
                cur_sec += dt_sec
                # msg を fluidsynth に送る
                if msg.type == "set_tempo":
                    tempo_us = msg.tempo
                elif msg.type == "program_change":
                    try:
                        fs.program_change(msg.channel, msg.program)
                    except Exception:
                        pass
                elif msg.type == "note_on":
                    if msg.velocity > 0:
                        fs.noteon(msg.channel, msg.note, msg.velocity)
                    else:
                        fs.noteoff(msg.channel, msg.note)
                elif msg.type == "note_off":
                    fs.noteoff(msg.channel, msg.note)
                elif msg.type == "control_change":
                    try:
                        fs.cc(msg.channel, msg.control, msg.value)
                    except Exception:
                        pass
            # 最後のリバーブ残響分
            tail = fs.get_samples(int(1.5 * SR))
            end = min(cursor_frame + len(tail), len(buf))
            buf[cursor_frame:end] = tail[: end - cursor_frame]
            fs.delete()

            # ステレオを float モノに
            stereo = buf.reshape(-1, 2).astype(np.float32) / 32768.0
            mono = stereo.mean(axis=1)
            return mono
        except Exception as e:
            self._log(f"  FluidSynth 失敗: {e}", -1)
            return None

    # ---- ブレンド＆マスタリング ---------------------------

    def _blend(self, synth_audio, orig_audio, sr, ratio):
        """合成音と原曲をクロスブレンド (ratio=1.0 で原曲のみ)"""
        n = max(len(synth_audio), len(orig_audio))
        a = np.zeros(n, dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)
        a[:len(synth_audio)] = synth_audio
        b[:len(orig_audio)] = orig_audio
        # RMS を揃える
        rms_a = np.sqrt(np.mean(a**2) + 1e-9)
        rms_b = np.sqrt(np.mean(b**2) + 1e-9)
        b *= rms_a / rms_b
        return (1 - ratio) * a + ratio * b

    def _master(self, audio):
        """簡易マスタリング: ソフトクリップ + ハイパスで低域ノイズ除去"""
        if len(audio) == 0:
            return audio.astype(np.float32)
        # ハイパス 30Hz
        nyq = SR / 2
        b, a = butter(2, 30 / nyq, btype="high")
        audio = filtfilt(b, a, audio).astype(np.float32)
        # ノーマライズ + ソフトクリップ
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        audio = np.tanh(audio * 1.1) * 0.92
        return audio.astype(np.float32)

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
        # テンポが検出できなかった場合は120BPMをデフォルトとする
        if tempo_val <= 0:
            tempo_val = 120.0
        return tempo_val, beat_times

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

    # ---- 15楽器シンセサイザー --------------------------------

    def _tone_piano(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        sig = sum((0.75**h)*np.sin(2*np.pi*hz*(h+1)*t) for h in range(8) if hz*(h+1)<SR/2)
        return sig * np.exp(-t*(1.5+hz/600)) * self._env(t,0.003,0.08,0.5,0.12) * (vel/127)*0.25

    def _tone_e_piano(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        mod = np.sin(2*np.pi*hz*t) * (vel/127) * 3
        sig = np.sin(2*np.pi*hz*t + mod) + 0.3*np.sin(2*np.pi*hz*2*t)*np.exp(-t*4)
        return sig * np.exp(-t*2) * self._env(t,0.002,0.1,0.4,0.15) * (vel/127)*0.22

    def _tone_glockenspiel(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(max(dur, 0.8))
        sig = (np.sin(2*np.pi*hz*t) + 0.6*np.sin(2*np.pi*hz*2.76*t)
               + 0.3*np.sin(2*np.pi*hz*5.4*t) + 0.1*np.sin(2*np.pi*hz*8.93*t))
        return sig * np.exp(-t*3) * self._env(t,0.001,0.02,0.3,0.2) * (vel/127)*0.15

    def _tone_organ(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        sig = (0.8*np.sin(2*np.pi*hz*0.5*t) + np.sin(2*np.pi*hz*t)
               + 0.6*np.sin(2*np.pi*hz*1.5*t) + 0.8*np.sin(2*np.pi*hz*2*t)
               + 0.3*np.sin(2*np.pi*hz*3*t) + 0.5*np.sin(2*np.pi*hz*4*t))
        return sig/4 * self._env(t,0.01,0.02,0.9,0.05) * (vel/127)*0.18

    def _tone_guitar_nylon(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        sig = sum((0.7**h)*np.sin(2*np.pi*hz*(h+1)*t) for h in range(6) if hz*(h+1)<SR/2)
        return sig * np.exp(-t*2.5) * self._env(t,0.002,0.06,0.4,0.08) * (vel/127)*0.20

    def _tone_guitar_clean(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        sig = sum((0.65**h)*np.sin(2*np.pi*hz*(h+1)*t+h*0.2) for h in range(8) if hz*(h+1)<SR/2)
        return sig * np.exp(-t*2.0) * self._env(t,0.001,0.04,0.45,0.1) * (vel/127)*0.18

    def _tone_bass(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        sig = np.sin(2*np.pi*hz*t) + 0.5*np.sin(2*np.pi*hz*2*t) + 0.2*np.sin(2*np.pi*hz*3*t)
        return sig * np.exp(-t*1.4) * self._env(t,0.008,0.05,0.75,0.1) * (vel/127)*0.35

    def _tone_violin(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        vib = 1 + 0.004*np.sin(2*np.pi*5.8*t)
        sig = sum(((-1)**h*0.7**h)*np.sin(2*np.pi*hz*(h+1)*vib*t) for h in range(6) if hz*(h+1)<SR/2)
        return sig * self._env(t,0.05,0.08,0.85,0.12) * (vel/127)*0.20

    def _tone_viola(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        vib = 1 + 0.003*np.sin(2*np.pi*5.2*t)
        sig = sum(((-1)**h*0.75**h)*np.sin(2*np.pi*hz*(h+1)*vib*t) for h in range(5) if hz*(h+1)<SR/2)
        return sig * self._env(t,0.06,0.1,0.8,0.15) * (vel/127)*0.18

    def _tone_cello(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        vib = 1 + 0.003*np.sin(2*np.pi*4.5*t)
        sig = sum((0.8**h)*np.sin(2*np.pi*hz*(h+1)*vib*t) for h in range(7) if hz*(h+1)<SR/2)
        return sig * self._env(t,0.04,0.1,0.8,0.15) * (vel/127)*0.22

    def _tone_strings(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        vib = 1 + 0.003*np.sin(2*np.pi*5*t)
        s1 = sum((0.7**h)*np.sin(2*np.pi*hz*(h+1)*vib*t) for h in range(5) if hz*(h+1)<SR/2)
        s2 = sum((0.7**h)*np.sin(2*np.pi*hz*1.002*(h+1)*vib*t) for h in range(5) if hz*1.002*(h+1)<SR/2)
        s3 = sum((0.7**h)*np.sin(2*np.pi*hz*0.998*(h+1)*vib*t) for h in range(5) if hz*0.998*(h+1)<SR/2)
        return (s1+s2+s3)/3 * self._env(t,0.12,0.15,0.7,0.2) * (vel/127)*0.15

    def _tone_choir(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        vib = 1 + 0.005*np.sin(2*np.pi*5.5*t)
        sig = (np.sin(2*np.pi*hz*vib*t) + 0.5*np.sin(2*np.pi*hz*2*vib*t)
               + 0.3*np.sin(2*np.pi*hz*3*vib*t) + 0.4*np.sin(2*np.pi*hz*1.003*vib*t)
               + 0.4*np.sin(2*np.pi*hz*0.997*vib*t))
        return sig/3 * self._env(t,0.2,0.15,0.65,0.25) * (vel/127)*0.12

    def _tone_trumpet(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        vib = 1 + 0.002*np.sin(2*np.pi*5.5*t)
        sig = (np.sin(2*np.pi*hz*vib*t) + 0.8*np.sin(2*np.pi*hz*2*vib*t)
               + 0.6*np.sin(2*np.pi*hz*3*vib*t) + 0.5*np.sin(2*np.pi*hz*4*vib*t)
               + 0.3*np.sin(2*np.pi*hz*5*vib*t))
        return sig/3 * self._env(t,0.03,0.06,0.85,0.08) * (vel/127)*0.20

    def _tone_flute(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        vib = 1 + 0.003*np.sin(2*np.pi*5.5*t)
        sig = np.sin(2*np.pi*hz*vib*t) + 0.1*np.sin(2*np.pi*hz*2*vib*t)
        sig += np.random.randn(len(t)) * 0.02 * np.exp(-t*3)
        return sig * self._env(t,0.06,0.1,0.8,0.15) * (vel/127)*0.18

    def _tone_pad(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69); t = self._t(dur)
        s1 = np.sin(2*np.pi*hz*t)
        s2 = np.sin(2*np.pi*hz*1.003*t)
        s3 = np.sin(2*np.pi*hz*0.997*t)
        return (s1+s2+s3)/3 * self._env(t,0.15,0.2,0.6,0.3) * (vel/127)*0.12

    TONE_FN = {
        'piano':'_tone_piano', 'e_piano':'_tone_e_piano',
        'glockenspiel':'_tone_glockenspiel', 'organ':'_tone_organ',
        'guitar_nylon':'_tone_guitar_nylon', 'guitar_clean':'_tone_guitar_clean',
        'bass':'_tone_bass', 'violin':'_tone_violin', 'viola':'_tone_viola',
        'cello':'_tone_cello', 'strings':'_tone_strings', 'choir':'_tone_choir',
        'trumpet':'_tone_trumpet', 'flute':'_tone_flute', 'pad':'_tone_pad',
    }

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

    def _save_midi(self, parts, drums, tempo, path):
        mid = MidiFile(type=1, ticks_per_beat=480)
        tpb = 480
        us = int(60_000_000 / tempo)
        def s2t(s): return int(s * tempo / 60 * tpb)
        tt = MidiTrack()
        tt.append(MetaMessage('set_tempo', tempo=us, time=0))
        mid.tracks.append(tt)
        for name, evts in parts.items():
            if not evts:
                continue
            ch, prog = self.MIDI_MAP[name]
            evs = [(0, Message('program_change', channel=ch, program=prog, time=0))]
            for (t, dur, midi, vel) in evts:
                note = int(np.clip(midi, 0, 127))
                v = min(max(vel, 1), 127)
                t0, t1 = s2t(t), s2t(t + dur)
                evs.append((t0, Message('note_on', channel=ch, note=note, velocity=v, time=0)))
                evs.append((t1, Message('note_off', channel=ch, note=note, velocity=0, time=0)))
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
        if devs:
            dtrk = MidiTrack()
            devs.sort(key=lambda x: x[0])
            prev = 0
            for tick, msg in devs:
                msg.time = max(0, tick - prev)
                dtrk.append(msg)
                prev = tick
            mid.tracks.append(dtrk)
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

        self.root.title("耳コピ自動生成ツール v7.0")
        self.root.geometry("640x700")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self._out_dir  = tk.StringVar(value=str(Path.home() / "Desktop"))
        self._status   = tk.StringVar(value="MP3ファイルをドロップしてください")
        self._progress = tk.DoubleVar(value=0)
        self._mode     = tk.StringVar(value="ai")
        self._blend    = tk.DoubleVar(value=0.0)
        self._busy     = False

        self._build()

    # ---- UI構築 -----------------------------------------

    def _build(self):
        r = self.root

        # タイトル
        tk.Label(r, text="耳コピ自動生成ツール v7.0",
                 font=("Helvetica", 20, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(pady=(20, 4))
        tk.Label(r, text="Demucs × Basic Pitch × FluidSynth で原曲忠実な耳コピ",
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

        # モード選択
        mf = tk.LabelFrame(r, text="  モード  ",
                           bg=self.BG, fg=self.FG2,
                           font=("Helvetica", 9), bd=1, relief="flat")
        mf.pack(padx=30, pady=(14, 0), fill="x")
        modes = [
            ("ai",      "AI耳コピ (Demucs+BasicPitch) / 推奨"),
            ("blend",   "原曲ブレンド (AI耳コピ + 原曲ミックス)"),
            ("karaoke", "カラオケ (ボーカル除去のみ / 最高忠実度)"),
            ("classic", "Classic (v6.0軽量 / AI不要)"),
        ]
        for val, label in modes:
            tk.Radiobutton(
                mf, text=label, variable=self._mode, value=val,
                bg=self.BG, fg=self.FG, selectcolor=self.BG2,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=("Helvetica", 9), anchor="w",
                command=self._on_mode_change
            ).pack(anchor="w", padx=8, pady=1)

        # ブレンド比率スライダー
        self._blend_frame = tk.Frame(mf, bg=self.BG)
        self._blend_frame.pack(fill="x", padx=10, pady=(3, 6))
        tk.Label(self._blend_frame, text="原曲ブレンド比率:",
                 bg=self.BG, fg=self.FG2, font=("Helvetica", 8)
                 ).pack(side="left")
        self._blend_scale = tk.Scale(
            self._blend_frame, from_=0.0, to=1.0, resolution=0.05,
            orient="horizontal", variable=self._blend,
            bg=self.BG, fg=self.FG, troughcolor=self.BG2,
            highlightthickness=0, length=280, showvalue=True,
            activebackground=self.ACCENT
        )
        self._blend_scale.pack(side="left", padx=(6, 0))
        self._blend_frame.pack_forget()  # ai モードでは非表示

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

    def _on_mode_change(self):
        if self._mode.get() == "blend":
            self._blend_frame.pack(fill="x", padx=10, pady=(3, 6))
        else:
            self._blend_frame.pack_forget()

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

        mode = self._mode.get()
        blend = float(self._blend.get()) if mode == "blend" else 0.0

        def worker():
            engine = EarCopyEngine(
                on_progress=self._on_prog, mode=mode, blend_ratio=blend
            )
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
# CLIモード
# =====================================================

def _run_cli(args):
    """コマンドライン引数で直接処理を実行する"""
    import argparse
    parser = argparse.ArgumentParser(
        description="耳コピ自動生成ツール v7.0 - Demucs×BasicPitch×FluidSynth"
    )
    parser.add_argument("input", help="入力音楽ファイル（MP3/WAV/M4A/FLAC）")
    parser.add_argument("-o", "--output", help="出力MP3パス（省略時: 同じフォルダに 耳コピ_*.mp3）")
    parser.add_argument("--mode", choices=["ai", "blend", "karaoke", "classic"],
                        default="ai", help="処理モード (デフォルト: ai)")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="blend モード時の原曲比率 0.0-1.0 (デフォルト: 0.0)")
    opts = parser.parse_args(args)

    inp = Path(opts.input)
    if not inp.exists():
        print(f"エラー: ファイルが見つかりません: {inp}")
        sys.exit(1)

    if opts.output:
        out = opts.output
    else:
        out = str(inp.parent / f"耳コピ_{inp.stem}.mp3")

    print(f"入力: {inp}")
    print(f"出力: {out}")
    print(f"モード: {opts.mode}")
    print()

    engine = EarCopyEngine(mode=opts.mode, blend_ratio=opts.blend)
    ok, result = engine.process(str(inp), out)

    if ok:
        midi_path = str(Path(out).with_suffix(".mid"))
        print(f"\n完了！")
        print(f"  MP3: {out}")
        if Path(midi_path).exists():
            print(f"  MIDI: {midi_path}")
    else:
        print(f"\nエラー:\n{result}")
        sys.exit(1)


# =====================================================
# エントリーポイント
# =====================================================

if __name__ == "__main__":
    try:
        # コマンドライン引数があればCLIモード、なければGUIモード
        if len(sys.argv) > 1:
            _run_cli(sys.argv[1:])
        else:
            _init_gui()
            App().run()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 50)
        print("エラーが発生しました。")
        print("上のメッセージをコピーして開発者に報告してください。")
        print("=" * 50)
        try:
            input("Enterキーで終了...")
        except EOFError:
            pass
