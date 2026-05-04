#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
耳コピ自動生成ツール v7.0
MP3をドラッグ&ドロップするだけで100% MIDI再合成・原盤不使用の耳コピ音源を自動作成

【著作権・原盤権ポリシー】
  本ツールは原曲音源を再配布しません。MIDI採譜後にサンプル音源 (SoundFont) で
  再合成した「自分の演奏」のみを出力します。Demucs で分離した原音ステムは
  MIDI 採譜の入力と評価スコア算出にのみ使用し、最終 WAV/MP3 には一切混合しません。

v7.0 の革新:
  - Demucs (Meta): ボーカル/ドラム/ベース/その他の AI ステム分離 (採譜入力専用)
  - Basic Pitch (Spotify): SOTA 多声部ポリフォニック MIDI 採譜
  - FluidSynth + SoundFont: 本物のサンプル音源による100% 再合成
  - キー推定＆スケールスナップ: ボーカル音程のAI自動補正
  - 出力品質チェック: 長さ/音量/スペクトル/クリッピングの自動検証

必要環境: Python 3.8+
初回起動時に依存パッケージを自動インストールします（AIモード時は約1-2GB）

使い方:
  GUI:  python mimikopi.py
  CLI:  python mimikopi.py input.mp3 [-o output.mp3] [--mode MODE]
        MODE: ai (Demucs+BasicPitch / デフォルト)
              ai_inst (AI耳コピ・ガイドメロディなし / カラオケ伴奏向け)
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

# =====================================================
# Transformer 採譜精製モジュール (torch オプション)
# =====================================================

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- MIDI トークン定数 (4-token グループ: TIME NOTE DUR VEL) ---
_T_PAD      = 0
_T_BOS      = 1
_T_EOS      = 2
_T_TIME_OFF = 3      # TIME: 0–500 (10ms 刻み, 最大 5000ms)
_T_NOTE_OFF = 504    # NOTE: 0–127 (MIDI ノート番号)
_T_DUR_OFF  = 632    # DUR:  0–200 (10ms 刻み, 最大 2000ms)
_T_VEL_OFF  = 833    # VEL:  0–127 (MIDI ベロシティ)
_T_VOCAB    = 961    # 総語彙数


def notes_to_tokens(notes):
    """[(onset_s, dur_s, midi, vel), ...] → List[int]  (BOS + 4tok*N + EOS)"""
    import numpy as _np
    toks = [_T_BOS]
    for (t, d, m, v) in notes:
        toks.extend([
            _T_TIME_OFF + int(_np.clip(round(t * 100), 0, 500)),
            _T_NOTE_OFF + int(_np.clip(m, 0, 127)),
            _T_DUR_OFF  + int(_np.clip(round(d * 100), 0, 200)),
            _T_VEL_OFF  + int(_np.clip(v, 0, 127)),
        ])
    toks.append(_T_EOS)
    return toks


def tokens_to_notes(toks):
    """List[int] → [(onset_s, dur_s, midi, vel), ...]  (BOS/EOS/PAD を除外)"""
    body = [t for t in toks if t not in (_T_PAD, _T_BOS, _T_EOS)]
    notes = []
    for i in range(0, len(body) - 3, 4):
        t_tok, n_tok, d_tok, v_tok = body[i], body[i+1], body[i+2], body[i+3]
        onset = (t_tok - _T_TIME_OFF) / 100.0
        midi  = n_tok - _T_NOTE_OFF
        dur   = (d_tok - _T_DUR_OFF) / 100.0
        vel   = v_tok - _T_VEL_OFF
        if (0.0 <= onset and 0.0 < dur <= 20.0
                and 0 <= midi <= 127 and 0 <= vel <= 127):
            notes.append((onset, dur, int(midi), int(vel)))
    return notes


def corrupt_notes(notes, pitch_jitter=2, time_jitter=0.03, dropout=0.10, rng=None):
    """学習データ生成: ノートにランダムノイズを加えて汚染する。
    pitch_jitter: ±半音数  time_jitter: ±秒  dropout: 削除確率
    """
    import numpy as _np
    if rng is None:
        rng = _np.random.default_rng()
    corrupted = []
    for (t, d, m, v) in notes:
        if rng.random() < dropout:
            continue
        dm = int(rng.integers(-pitch_jitter, pitch_jitter + 1))
        dt = float(rng.uniform(-time_jitter, time_jitter))
        corrupted.append((
            max(0.0, t + dt),
            d,
            int(_np.clip(m + dm, 0, 127)),
            v,
        ))
    return sorted(corrupted, key=lambda x: x[0])


if HAS_TORCH:
    class MusicTransformer(nn.Module):
        """4 層 Transformer Encoder: ノートシーケンスのデノイジング精製用。
        入力: トークン列 (B, T)  出力: ロジット (B, T, VOCAB)
        """
        def __init__(self, vocab=_T_VOCAB, d_model=128, nhead=4,
                     num_layers=4, dim_ff=256, max_len=512, dropout=0.1):
            super().__init__()
            self.embed   = nn.Embedding(vocab, d_model, padding_idx=_T_PAD)
            self.pos_enc = nn.Embedding(max_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_ff, dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.head    = nn.Linear(d_model, vocab)

        def forward(self, x, src_key_padding_mask=None):
            """x: (B, T) int → (B, T, VOCAB) float"""
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.embed(x) + self.pos_enc(pos)
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
            return self.head(h)
else:
    # torch 未インストール時はダミークラスを定義
    class MusicTransformer:  # type: ignore[no-redef]
        """torch 未インストール時のスタブ。_transformer_refine は no-op になる。"""
        def __init__(self, **_):
            pass


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
    except ImportError:
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

    # 3. アセットディレクトリを検索
    if _ASSETS_DIR.exists():
        exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
        for p in _ASSETS_DIR.rglob(exe):
            if p.is_file():
                _save_ffmpeg(str(p))
                return str(p)

    # 4. よくある場所を自動検索（Windows）
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

    # 5. 自動ダウンロード・インストール
    auto = _download_ffmpeg()
    if auto:
        return auto

    # 6. ユーザーに手動選択させる（GUIダイアログ）
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


def _download_ffmpeg(log=print):
    """ffmpeg を自動ダウンロード・インストールする"""
    system = platform.system()
    _ASSETS_DIR.mkdir(exist_ok=True)
    import urllib.request

    if system == "Windows":
        urls = [
            ("FFmpeg (BtbN/GitHub)",
             "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"),
        ]
        for name, url in urls:
            log(f"  ffmpeg をダウンロード中 ({name})...")
            try:
                import zipfile
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EarCopyTool)"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) < 1_000_000:
                    raise RuntimeError(f"ファイルサイズが小さすぎます ({len(data)} bytes)")
                dest_zip = _ASSETS_DIR / "ffmpeg.zip"
                dest_zip.write_bytes(data)
                with zipfile.ZipFile(dest_zip) as zf:
                    zf.extractall(_ASSETS_DIR)
                dest_zip.unlink(missing_ok=True)
                for p in _ASSETS_DIR.rglob("ffmpeg.exe"):
                    if p.is_file():
                        _save_ffmpeg(str(p))
                        log(f"  ✓ ffmpeg: {p}")
                        return str(p)
                log(f"  ✗ {name}: ZIPにffmpegが含まれず")
            except Exception as e:
                log(f"  ✗ {name} 取得失敗: {e}")
        return None

    if system == "Linux":
        for mgr_cmd in [["apt-get", "install", "-y", "ffmpeg"], ["apt", "install", "-y", "ffmpeg"],
                         ["dnf", "install", "-y", "ffmpeg"], ["pacman", "-S", "--noconfirm", "ffmpeg"]]:
            mgr = mgr_cmd[0]
            if not shutil.which(mgr):
                continue
            log(f"  ffmpeg をインストール中 ({mgr})...")
            try:
                subprocess.check_call(mgr_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                ff = shutil.which("ffmpeg")
                if ff:
                    log(f"  ✓ ffmpeg: {ff}")
                    return ff
            except (subprocess.CalledProcessError, PermissionError):
                try:
                    subprocess.check_call(["sudo"] + mgr_cmd, timeout=120)
                    ff = shutil.which("ffmpeg")
                    if ff:
                        log(f"  ✓ ffmpeg: {ff}")
                        return ff
                except Exception:
                    pass
            except Exception:
                pass
        urls = [
            ("FFmpeg (johnvansickle)",
             "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"),
        ]
        for name, url in urls:
            log(f"  ffmpeg をダウンロード中 ({name})...")
            try:
                import tarfile
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EarCopyTool)"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                if len(data) < 1_000_000:
                    raise RuntimeError(f"ファイルサイズが小さすぎます ({len(data)} bytes)")
                dest_tar = _ASSETS_DIR / "ffmpeg.tar.xz"
                dest_tar.write_bytes(data)
                with tarfile.open(dest_tar) as tf:
                    tf.extractall(_ASSETS_DIR)
                dest_tar.unlink(missing_ok=True)
                for p in _ASSETS_DIR.rglob("ffmpeg"):
                    if p.is_file() and not p.suffix:
                        p.chmod(0o755)
                        _save_ffmpeg(str(p))
                        log(f"  ✓ ffmpeg: {p}")
                        return str(p)
                log(f"  ✗ {name}: アーカイブにffmpegが含まれず")
            except Exception as e:
                log(f"  ✗ {name} 取得失敗: {e}")
        return None

    if system == "Darwin":
        if shutil.which("brew"):
            log("  ffmpeg をインストール中 (Homebrew)...")
            try:
                subprocess.check_call(["brew", "install", "ffmpeg"], timeout=300)
                ff = shutil.which("ffmpeg")
                if ff:
                    log(f"  ✓ ffmpeg: {ff}")
                    return ff
            except Exception as e:
                log(f"  ✗ Homebrew install 失敗: {e}")
        log("  ffmpeg が見つかりません。以下を実行してください:")
        log("    brew install ffmpeg")
        return None

    return None


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

_SF2_DOWNLOADS = [
    # (表示名, URL, 想定サイズMB, 拡張子)
    # 第1優先: GeneralUser GS v1.471 (高品質、コンパクト、商用可)
    ("GeneralUser GS v1.471 (GitHub/ROCKNIX)",
     "https://github.com/ROCKNIX/generaluser-gs/raw/main/GeneralUser%20GS%20v1.471.sf2",
     30, "sf2"),
    ("GeneralUser GS v1.471 (GitHub/JELOS)",
     "https://github.com/JustEnoughLinuxOS/generaluser-gs/raw/main/GeneralUser%20GS%20v1.471.sf2",
     30, "sf2"),
    ("GeneralUser GS v1.471 (Musical Artifacts)",
     "https://musical-artifacts.com/artifacts/1390/GeneralUser_GS_v1.471.sf2",
     30, "sf2"),
    # 第2優先: MuseScore / MS Basic SF3 (公式OSS、多サンプル)
    ("MuseScore General SF3",
     "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3",
     38, "sf3"),
    ("MS Basic SF3 (MuseScore GitHub)",
     "https://github.com/musescore/MuseScore/raw/master/share/sound/MS%20Basic.sf3",
     38, "sf3"),
    # 第3優先: FluidR3_GM (現行、フォールバック)
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

# 高品質ピアノ専用 SoundFont (CC-BY 3.0, Alexander Holm)
# Salamander Grand Piano: Yamaha C5 からサンプリングした 48kHz 16bit ピアノ
_PIANO_SF2_DOWNLOADS = [
    ("Salamander Grand Piano v3 SF2 (FreePats)",
     "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPianoV3+20161209_48khz24bit.sf2",
     400, "sf2"),
]

# instruments.json ベースの楽器→SF2マッピング設定パス
_INSTRUMENTS_JSON = Path(__file__).parent / "instruments.json"

_SF2_QUALITY_KEYWORDS = [
    ("generaluser",     100),
    ("musescore_gen",    95),
    ("ms basic",         93),
    ("ms_basic",         93),
    ("sgm",              90),
    ("timbres",          88),
    ("compifont",        85),
    ("fluidr3",          70),
    ("default",          30),
]


def _sf2_quality_score(path: str) -> int:
    name = Path(path).name.lower()
    base_score = 0
    for kw, score in _SF2_QUALITY_KEYWORDS:
        if kw in name:
            base_score = max(base_score, score)
    try:
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        if size_mb >= 200:
            base_score += 15
        elif size_mb >= 100:
            base_score += 10
        elif size_mb >= 30:
            base_score += 5
    except Exception:
        pass
    return base_score


def _find_sf2():
    """SoundFont (.sf2/.sf3) のパスを解決する"""
    # 0. 環境変数で明示指定された場合は最優先
    env_sf2 = os.environ.get("MIMIKOPI_SF2", "").strip()
    if env_sf2 and Path(env_sf2).exists():
        return env_sf2
    # 1. 前回保存したパス
    if _SF2_CONFIG.exists():
        try:
            saved = json.loads(_SF2_CONFIG.read_text())
            p = saved.get("path", "")
            if p and Path(p).exists():
                return p
        except Exception:
            pass

    # 2. アセットディレクトリを検索（品質スコア最高のものを選択）
    if _ASSETS_DIR.exists():
        local_fonts = []
        for ext in ("*.sf2", "*.sf3"):
            local_fonts.extend(_ASSETS_DIR.glob(ext))
        if local_fonts:
            best = max(local_fonts, key=lambda p: _sf2_quality_score(str(p)))
            return str(best)

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
    found = [c for c in candidates if Path(c).exists()]
    if found:
        best = max(found, key=lambda c: _sf2_quality_score(c))
        _save_sf2(best)
        return best
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

            # SF2/SF3 ヘッダ検証 (RIFF...sfbk)
            is_valid_sf = (len(data) > 16
                           and data[:4] == b"RIFF"
                           and b"sfbk" in data[:64])
            if is_valid_sf:
                _save_sf2(str(dest))
                log(f"  ✓ {dest.name}")
                return str(dest)
            dest.unlink(missing_ok=True)
            log(f"  ✗ {name}: SoundFontフォーマット不正")
        except Exception as e:
            log(f"  ✗ {name} 取得失敗: {e}")
            continue
    return None


def _find_piano_sf2():
    """ピアノ専用 SoundFont を探す (Salamander 優先)"""
    for ext in ("*.sf2", "*.sf3"):
        for p in _ASSETS_DIR.glob(ext):
            if "salamander" in p.name.lower() or "piano" in p.name.lower():
                return str(p)
    return None


def _download_piano_sf2(log=print):
    """Salamander Grand Piano SF2 をダウンロード (CC-BY 3.0, Alexander Holm)"""
    _ASSETS_DIR.mkdir(exist_ok=True)
    import urllib.request
    for name, url, size, ext in _PIANO_SF2_DOWNLOADS:
        log(f"  ピアノ SoundFont をダウンロード中 ({name}, 約{size}MB)...")
        try:
            fname = f"SalamanderGrandPiano.{ext}"
            dest = _ASSETS_DIR / fname
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (EarCopyTool)"
            })
            with urllib.request.urlopen(req, timeout=120) as r:
                ct = r.headers.get("Content-Type", "").lower()
                if "text/html" in ct:
                    raise RuntimeError(f"HTML応答 (Content-Type: {ct})")
                data = r.read()
            if len(data) < 1_000_000:
                raise RuntimeError(f"サイズ不正 ({len(data)} bytes)")
            dest.write_bytes(data)
            if len(data) > 16 and data[:4] == b"RIFF":
                log(f"  ✓ {dest.name}")
                return str(dest)
            dest.unlink(missing_ok=True)
        except Exception as e:
            log(f"  ✗ {name}: {e}")
    return None


def _load_instrument_config():
    """instruments.json を読み込み、楽器グループ→SF2 マッピングを返す"""
    if not _INSTRUMENTS_JSON.exists():
        return None
    try:
        return json.loads(_INSTRUMENTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_sf2_for_group(group_cfg, config):
    """楽器グループ設定から使用するSF2パスを解決する"""
    sf2_key = group_cfg.get("sf2_key", "general")
    fallback_key = group_cfg.get("fallback_sf2_key", "general")

    if sf2_key == "salamander_piano":
        piano_sf2 = _find_piano_sf2()
        if piano_sf2:
            return piano_sf2
        if fallback_key != sf2_key:
            return _find_sf2()
    return _find_sf2()


def _generate_license_readme():
    """使用中の SoundFont / IR のライセンス情報を README_LICENSES.txt に出力する"""
    config = _load_instrument_config()
    if not config:
        return
    lines = [
        "# Noion - 使用ライセンス一覧",
        "# 自動生成ファイル — 手動編集不要",
        "",
        "## SoundFont ライセンス",
        "",
    ]
    for key, info in config.get("soundfonts", {}).items():
        lines.append(f"### {info.get('description', key)}")
        lines.append(f"- ライセンス: {info.get('license', '不明')}")
        lines.append(f"- 作者: {info.get('author', '不明')}")
        lines.append(f"- URL: {info.get('url', '')}")
        if info.get("credit_required"):
            lines.append("- ⚠ クレジット表記が必要")
        lines.append("")
    lines.append("## インパルスレスポンス (リバーブ)")
    lines.append("")
    for key, info in config.get("impulse_responses", {}).items():
        lines.append(f"### {info.get('description', key)}")
        lines.append(f"- ライセンス: {info.get('license', 'CC0')}")
        lines.append("")
    lines.append("---")
    lines.append("本ツールは原曲音源を再配布しません。MIDI採譜後にサンプル音源で")
    lines.append("再合成した出力のみを生成します。")
    readme_path = Path(__file__).parent / "README_LICENSES.txt"
    try:
        readme_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


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



# =====================================================
# FluidSynth CLI バイナリ自動セットアップ
# =====================================================

_FLUIDSYNTH_CONFIG = Path(__file__).parent / ".fluidsynth_path.json"
_FLUIDSYNTH_DL = {
    "Windows": (
        "https://github.com/FluidSynth/fluidsynth/releases/download/v2.4.6/fluidsynth-2.4.6-win10-x64.zip",
        "fluidsynth.exe"
    ),
}


def _find_fluidsynth_cli():
    """FluidSynth CLI バイナリを検出する"""
    # 1. PATH にある場合
    fs = shutil.which("fluidsynth")
    if fs:
        return fs

    # 2. 前回保存パス
    if _FLUIDSYNTH_CONFIG.exists():
        try:
            saved = json.loads(_FLUIDSYNTH_CONFIG.read_text())
            p = saved.get("path", "")
            if p and Path(p).exists():
                return p
        except Exception:
            pass

    # 3. _assets 内を検索
    if _ASSETS_DIR.exists():
        for p in _ASSETS_DIR.rglob("fluidsynth*"):
            if p.is_file() and p.suffix in (".exe", ""):
                _save_fluidsynth_path(str(p))
                return str(p)

    return None


def _save_fluidsynth_path(path):
    try:
        _FLUIDSYNTH_CONFIG.write_text(json.dumps({"path": str(path)}))
    except Exception:
        pass


def _download_fluidsynth(log=print):
    """Windows 用 FluidSynth バイナリを GitHub からダウンロードする"""
    system = platform.system()
    if system not in _FLUIDSYNTH_DL:
        if system == "Linux":
            for mgr_cmd in [["apt-get", "install", "-y", "fluidsynth"],
                             ["apt", "install", "-y", "fluidsynth"],
                             ["dnf", "install", "-y", "fluidsynth"],
                             ["pacman", "-S", "--noconfirm", "fluidsynth"]]:
                mgr = mgr_cmd[0]
                if not shutil.which(mgr):
                    continue
                log(f"  FluidSynth をインストール中 ({mgr})...")
                try:
                    subprocess.check_call(mgr_cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, timeout=120)
                    fs = shutil.which("fluidsynth")
                    if fs:
                        _save_fluidsynth_path(fs)
                        log(f"  ✓ FluidSynth: {fs}")
                        return fs
                except (subprocess.CalledProcessError, PermissionError):
                    try:
                        subprocess.check_call(["sudo"] + mgr_cmd, timeout=120)
                        fs = shutil.which("fluidsynth")
                        if fs:
                            _save_fluidsynth_path(fs)
                            log(f"  ✓ FluidSynth: {fs}")
                            return fs
                    except Exception:
                        pass
                except Exception:
                    pass
        elif system == "Darwin":
            if shutil.which("brew"):
                log("  FluidSynth をインストール中 (Homebrew)...")
                try:
                    subprocess.check_call(["brew", "install", "fluid-synth"], timeout=300)
                    fs = shutil.which("fluidsynth")
                    if fs:
                        _save_fluidsynth_path(fs)
                        log(f"  ✓ FluidSynth: {fs}")
                        return fs
                except Exception as e:
                    log(f"  ✗ Homebrew install 失敗: {e}")
        log(f"  FluidSynth: {system} で自動インストールに失敗")
        if system == "Darwin":
            log("    → brew install fluid-synth")
        else:
            log("    → sudo apt install fluidsynth")
        return None

    url, exe_name = _FLUIDSYNTH_DL[system]
    _ASSETS_DIR.mkdir(exist_ok=True)
    log(f"  FluidSynth をダウンロード中 (~15MB)...")

    import urllib.request
    import zipfile
    try:
        dest_zip = _ASSETS_DIR / "fluidsynth.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EarCopyTool)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) < 500_000:
            raise RuntimeError(f"ダウンロードが小さすぎます ({len(data)} bytes)")
        dest_zip.write_bytes(data)

        # ZIP展開
        with zipfile.ZipFile(dest_zip) as zf:
            zf.extractall(_ASSETS_DIR)

        # fluidsynth.exe を再帰検索
        for p in _ASSETS_DIR.rglob(exe_name):
            if p.is_file():
                _save_fluidsynth_path(str(p))
                log(f"  ✓ FluidSynth: {p}")
                return str(p)
        log("  ✗ ZIPにFluidSynthバイナリが見つかりません")
    except Exception as e:
        log(f"  ✗ FluidSynth DL失敗: {e}")
    return None


def _ensure_fluidsynth_cli(log=print):
    """FluidSynth CLI が使える状態にする"""
    fs = _find_fluidsynth_cli()
    if fs:
        return fs
    return _download_fluidsynth(log)


# =====================================================
# 音楽分析エンジン
# =====================================================

SR = 44100
HOP = 512
SEMI = 2 ** (1 / 12)


class EarCopyEngine:
    """v7.0 - Demucs + Basic Pitch + FluidSynth を主軸とした 100% MIDI再合成エンジン。

    出力ポリシー:
      最終 WAV/MP3 は FluidSynth 合成音のみで構成し、Demucs で分離した
      原音ステムは一切混入しない (原盤権侵害の防止)。

    モード:
      - "ai"      : Demucs でステム分離 → Basic Pitch で多声部採譜 → FluidSynth 合成
                    （未セットアップ時は CQT+pyin フォールバック）
      - "ai_inst" : ai と同じパイプラインだがボーカルを完全に除去（カラオケ伴奏向け）
    """

    # Old 15-instrument map (before consolidation):
    # MIDI_MAP = {
    #     'piano': (0, 0), 'e_piano': (1, 4), 'glockenspiel': (2, 9),
    #     'organ': (3, 19), 'guitar_nylon': (4, 24), 'guitar_clean': (5, 27),
    #     'bass': (6, 33), 'violin': (7, 40), 'viola': (8, 41),
    #     'cello': (10, 42), 'strings': (11, 48), 'choir': (12, 52),
    #     'trumpet': (13, 56), 'flute': (14, 73), 'pad': (15, 89),
    # }
    MIDI_MAP = {
        'melody': (0, 0),       # Acoustic Grand Piano
        'chord': (1, 4),        # Electric Piano
        'bass': (2, 33),        # Electric Bass
        'pad': (3, 89),         # Pad (Warm)
        'decoration': (4, 9),   # Glockenspiel
        'sub_melody': (5, 42),  # Cello
    }
    GAIN = {
        'melody': 1.0, 'chord': 0.80, 'bass': 0.95,
        'pad': 0.45, 'decoration': 0.65, 'sub_melody': 0.85,
    }
    # 15-instrument names used internally by _ai_assign_parts / _smart_assign
    # before _consolidate_parts reduces them to the 5 roles in MIDI_MAP.
    _RAW_INST_NAMES = [
        'piano', 'e_piano', 'glockenspiel', 'organ',
        'guitar_nylon', 'guitar_clean', 'bass', 'violin', 'viola',
        'cello', 'strings', 'choir', 'trumpet', 'flute', 'pad',
    ]
    CHORD_IV = {
        'maj': [0,4,7], 'min': [0,3,7], 'dom7': [0,4,7,10],
        'min7': [0,3,7,10], 'maj7': [0,4,7,11], 'dim': [0,3,6],
        'aug': [0,4,8], 'sus4': [0,5,7],
    }
    NOTE_MIDI = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,
                 'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}

    def __init__(self, on_progress=None, mode="ai"):
        self._cb = on_progress
        self.mode = mode  # "ai" | "ai_inst"
        # 採譜精度モード: "high_recall" | "balanced" | "high_precision"
        self.transcription_quality = "balanced"

    def _log(self, msg, pct=None):
        if self._cb:
            self._cb(msg, pct)
        else:
            print(f"[{pct or '--':>3}%] {msg}")

    # ---- 採譜パラメータ設定 -----------------------------------

    def _get_transcription_cfg(self):
        """採譜モード別の閾値セットを返す。
        high_recall:    短音・弱音を逃さない（誤検出は増える）
        balanced:       デフォルト。プロレベル精度を目指す
        high_precision: 確信度の高いノートのみ（取りこぼしは増える）
        """
        cfgs = {
            'high_recall': {
                'onset_threshold': 0.20, 'onset_threshold_vocal': 0.22,
                'frame_threshold': 0.14, 'frame_threshold_vocal': 0.16,
                'min_note_ms': 18, 'min_note_ms_vocal': 20,
                'freq_min_vocal': 60.0, 'freq_max_vocal': 3200.0,
                'freq_min': 24.0, 'freq_max': 8000.0,
                'cqt_thr_delta': 24.0, 'cqt_thr_min': 0.5,
                'dur_floor_s': 0.02,
            },
            'balanced': {
                'onset_threshold': 0.25, 'onset_threshold_vocal': 0.28,
                'frame_threshold': 0.18, 'frame_threshold_vocal': 0.20,
                'min_note_ms': 20, 'min_note_ms_vocal': 25,
                'freq_min_vocal': 70.0, 'freq_max_vocal': 2800.0,
                'freq_min': 28.0, 'freq_max': 6000.0,
                'cqt_thr_delta': 24.0, 'cqt_thr_min': 0.8,
                'dur_floor_s': 0.02,
            },
            'high_precision': {
                'onset_threshold': 0.45, 'onset_threshold_vocal': 0.50,
                'frame_threshold': 0.30, 'frame_threshold_vocal': 0.35,
                'min_note_ms': 50, 'min_note_ms_vocal': 60,
                'freq_min_vocal': 90.0, 'freq_max_vocal': 2000.0,
                'freq_min': 40.0, 'freq_max': 4000.0,
                'cqt_thr_delta': 18.0, 'cqt_thr_min': 2.0,
                'dur_floor_s': 0.05,
            },
        }
        return cfgs.get(self.transcription_quality, cfgs['balanced'])

    # ---- ステム別前処理 ----------------------------------------

    def _preprocess_stem(self, audio, sr, stem_type):
        """ステムタイプごとに最適な前処理（正規化 + 帯域EQ）を行う。
        採譜精度向上のための純粋な信号処理。元音源は出力に使わない。
        """
        if audio is None or np.max(np.abs(audio)) < 1e-6:
            return audio
        # RMS 正規化: 採譜ライブラリの感度を均一化
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms > 1e-8:
            audio = audio / rms * 0.20
        nyq = sr / 2
        try:
            if stem_type == 'vocals':
                # ボーカル: 80Hz–8kHz バンドパス（ルーム・サブノイズ除去）
                b, a = butter(4, [min(80 / nyq, 0.99), min(8000 / nyq, 0.99)], btype='band')
                audio = filtfilt(b, a, audio).astype(np.float32)
            elif stem_type == 'other':
                # 伴奏: HPSS でハーモニック成分を強調（CQT精度向上）
                audio, _ = librosa.effects.hpss(audio, margin=2.0)
            elif stem_type == 'bass':
                # ベース: 300Hz ローパス（中高音ブリードを除去）
                b, a = butter(4, min(300 / nyq, 0.99), btype='low')
                audio = filtfilt(b, a, audio).astype(np.float32)
        except Exception:
            pass
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    # ---- MIDI 統計ログ -----------------------------------------

    def _log_midi_stats(self, label, notes):
        """採譜結果の診断ログを出力する。"""
        if not notes:
            self._log(f"  [{label}] ノートなし", -1)
            return
        durs = np.array([d for _, d, _, _ in notes], dtype=np.float32)
        vels = np.array([v for _, _, _, v in notes], dtype=np.float32)
        short_cnt = int(np.sum(durs < 0.10))
        clip_cnt = int(np.sum(vels >= 127))
        total = len(notes)
        self._log(
            f"  [{label}] {total}音符 | 平均長{np.mean(durs):.2f}s"
            f" | 短音{short_cnt}個({100*short_cnt//total}%)"
            f" | クリップ{clip_cnt}個({100*clip_cnt//total}%)",
            -1,
        )

    # ---- 重複・オーバーラップ ノート除去 -----------------------

    def _remove_overlapping_notes(self, notes):
        """同一ピッチで時間的に重なるノートを統合または末端カットする。
        同ピッチが再開始(onset)される場合は前ノートを直前で打ち切る。
        """
        if not notes:
            return notes
        by_pitch = {}
        for note in notes:
            t, dur, midi, vel = note
            by_pitch.setdefault(midi, []).append(note)
        result = []
        for midi, group in by_pitch.items():
            group = sorted(group, key=lambda x: x[0])
            cleaned = []
            for i, (t, dur, m, v) in enumerate(group):
                if cleaned:
                    prev_t, prev_dur, prev_m, prev_v = cleaned[-1]
                    overlap_end = prev_t + prev_dur
                    if t < overlap_end:
                        # 前ノートを今のノート開始直前でカット
                        new_dur = max(0.02, t - prev_t)
                        cleaned[-1] = (prev_t, new_dur, prev_m, prev_v)
                cleaned.append((t, dur, m, v))
            result.extend(cleaned)
        return sorted(result, key=lambda x: x[0])

    # ---- プロレベル後処理パイプライン ==========================

    def _refine_notes(self, notes, scale_set, tempo):
        """ノート後処理: スケール補正・極短ノート除去・隣接結合・跳躍抑制。"""
        if not notes:
            return notes
        # 1. 0.03秒未満のノートを削除
        notes = [(t, d, m, v) for t, d, m, v in notes if d >= 0.03]
        if not notes:
            return notes

        # 2. スケール外音を近い音に補正（スケール情報がある場合）
        if scale_set:
            refined = []
            for (t, d, m, v) in notes:
                pc = m % 12
                if pc not in scale_set:
                    for delta in [-1, 1, -2, 2]:
                        if (pc + delta) % 12 in scale_set:
                            m = m + delta
                            break
                refined.append((t, d, m, v))
            notes = refined

        # 3. 隣接する同一ピッチノートの結合（ギャップ30ms以内）
        notes = sorted(notes, key=lambda x: (x[2], x[0]))
        merged = []
        for note in notes:
            if merged:
                pt, pd, pm, pv = merged[-1]
                nt, nd, nm, nv = note
                gap = nt - (pt + pd)
                if nm == pm and 0 <= gap <= 0.03:
                    merged[-1] = (pt, nt + nd - pt, pm, max(pv, nv))
                    continue
            merged.append(note)
        notes = sorted(merged, key=lambda x: x[0])

        # 4. ±12半音以上の跳躍を抑制（隣接ノートとの差が大きすぎる場合、
        #    中間の音に丸める）
        if len(notes) > 1:
            clamped = [notes[0]]
            for i in range(1, len(notes)):
                t, d, m, v = notes[i]
                _, _, prev_m, _ = clamped[-1]
                diff = m - prev_m
                if abs(diff) > 12:
                    m = prev_m + (12 if diff > 0 else -12)
                    m = max(0, min(127, m))
                clamped.append((t, d, m, v))
            notes = clamped
        return notes

    def _rebuild_rhythm(self, notes, tempo):
        """リズム再構築: 16分音符グリッドへの軽量スナップ（50ms以内のみ）。"""
        if not notes or tempo <= 0:
            return notes
        beat_dur = 60.0 / max(tempo, 40)
        grid_step = beat_dur / 4  # 16分音符
        snap_limit = 0.05  # 50ms

        result = []
        for (t, dur, midi, vel) in notes:
            # スタート位置のスナップ（50ms以内のみ）
            nearest_grid = round(t / grid_step) * grid_step
            if abs(nearest_grid - t) <= snap_limit:
                qt = nearest_grid
            else:
                qt = t
            # デュレーションのグリッド合わせ（膨張なし）
            if dur >= grid_step:
                qdur = round(dur / grid_step) * grid_step
                qdur = max(qdur, grid_step)
                qdur = min(qdur, dur * 1.15)
            else:
                qdur = dur  # 短音は絶対に変えない
            result.append((qt, qdur, midi, vel))
        return result

    def _filter_notes_by_chord(self, notes, chords):
        """コード制約フィルタ: 各時刻のコードに合わない音を弱める/削除。
        コードトーンは保持、非コードトーンは velocity を下げる。"""
        if not notes or not chords:
            return notes
        result = []
        for (t, dur, midi, vel) in notes:
            # 該当時刻のコードを検索
            chord = None
            for ct, croot, cqual, cdur in chords:
                if ct <= t < ct + cdur:
                    chord = (croot, cqual)
                    break
            if chord is None:
                result.append((t, dur, midi, vel))
                continue
            root_name, quality = chord
            root_pc = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,
                       'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}.get(root_name, 0)
            chord_ivs = self.CHORD_IV.get(quality, [0, 4, 7])
            chord_pcs = set((root_pc + iv) % 12 for iv in chord_ivs)
            pc = midi % 12
            if pc in chord_pcs:
                result.append((t, dur, midi, vel))  # コードトーン: 保持
            else:
                # テンションとして扱う: velocity を 60% に
                result.append((t, dur, midi, int(vel * 0.6)))
        return result

    def _fill_harmony(self, notes, chords, tempo):
        """ノート密度補完: コードトーンを追加してスカスカな区間を埋める。
        既にノートがある時刻には追加しない。"""
        if not chords:
            return notes
        beat_dur = 60.0 / max(tempo, 40)
        existing_times = set(round(t / 0.05) for t, _, _, _ in notes)
        additions = []
        for (ct, root_name, quality, cdur) in chords:
            root_pc = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,
                       'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}.get(root_name, 0)
            chord_ivs = self.CHORD_IV.get(quality, [0, 4, 7])
            # この区間にノートが少ない場合のみ補完
            notes_in_range = [n for n in notes if ct <= n[0] < ct + cdur]
            if len(notes_in_range) >= 3:
                continue
            # アルペジオ的にコードトーンを追加
            step = beat_dur / 4
            for i, iv in enumerate(chord_ivs):
                fill_t = ct + step * i
                time_key = round(fill_t / 0.05)
                if time_key in existing_times:
                    continue
                midi_note = 60 + root_pc + iv  # オクターブ4
                additions.append((fill_t, min(step * 0.8, cdur - step * i), midi_note, 50))
                existing_times.add(time_key)
        return sorted(notes + additions, key=lambda x: x[0])

    # ---- Transformer 精製 ----------------------------------------

    def _transformer_refine(self, notes):
        """MusicTransformer でノートシーケンスを精製する（推論専用）。
        torch 未インストール / シーケンス過長 / エラー時は元ノートをそのまま返す。
        現在は未学習モデルのため no-op フォールバックが基本動作。
        """
        if not HAS_TORCH or not notes:
            return notes
        try:
            model = MusicTransformer()
            model.eval()
            toks = notes_to_tokens(notes)
            max_len = 512
            if len(toks) > max_len:
                return notes
            pad_len = max_len - len(toks)
            padded  = toks + [_T_PAD] * pad_len
            x    = torch.tensor([padded], dtype=torch.long)
            mask = (x == _T_PAD)
            with torch.no_grad():
                logits  = model(x, src_key_padding_mask=mask[0])
                pred    = logits[0].argmax(dim=-1).tolist()
            refined = tokens_to_notes(pred[:len(toks)])
            # 精製後に大幅にノートが減る場合は元を使う（未学習モデル対策）
            if len(refined) < len(notes) * 0.5:
                return notes
            return refined
        except Exception:
            return notes

    # ---- Sound Horizon 最適化 ------------------------------------

    def _thicken_chords(self, notes, chords):
        """コード内声部充填: 検出コードに対し不足する構成音を中声部に追加する。
        Sound Horizon スタイルの豊かなハーモニーを実現。
        """
        if not chords or not notes:
            return notes
        additions = []
        for (ct, root_name, quality, cdur) in chords:
            root_pc = self.NOTE_MIDI.get(root_name, 0)
            chord_ivs = self.CHORD_IV.get(quality, [0, 4, 7])
            present = [n for n in notes if ct <= n[0] < ct + cdur]
            present_pcs = set(n[2] % 12 for n in present)
            avg_vel = int(np.mean([n[3] for n in present])) if present else 55
            for iv in chord_ivs:
                target_pc = (root_pc + iv) % 12
                if target_pc not in present_pcs:
                    # 中声部 (MIDI 48–84) に追加
                    midi_note = 60 + target_pc
                    if midi_note > 84:
                        midi_note -= 12
                    if 48 <= midi_note <= 84:
                        additions.append((
                            ct + cdur * 0.05,
                            min(cdur * 0.7, 0.6),
                            midi_note,
                            int(avg_vel * 0.65),
                        ))
                        present_pcs.add(target_pc)
        return sorted(notes + additions, key=lambda x: x[0])

    def _extend_sustains(self, notes):
        """サスティン延長: 同一ピッチ間の短いギャップをレガートで埋める。
        Sound Horizon スタイルの流れるような旋律感を演出。
        """
        if not notes:
            return notes
        by_pitch = {}
        for n in notes:
            by_pitch.setdefault(n[2], []).append(n)
        result = list(notes)
        for midi, group in by_pitch.items():
            group = sorted(group, key=lambda x: x[0])
            for i in range(len(group) - 1):
                t, d, m, v = group[i]
                t_next = group[i + 1][0]
                gap = t_next - (t + d)
                # 50ms 以内のギャップはサスティンで埋める
                if 0.0 < gap <= 0.05:
                    idx = result.index(group[i])
                    if idx >= 0:
                        result[idx] = (t, d + gap * 0.9, m, v)
        return sorted(result, key=lambda x: x[0])

    def _detect_modulation(self, y, sr, beats):
        """転調検出: 楽曲を区間分割してキー変化を検出する。
        Returns: [(time_s, scale_set, scale_name), ...]  昇順
        """
        try:
            if beats is None or len(beats) < 8:
                ss, sn = self._estimate_key(y, sr)
                return [(0.0, ss, sn)]
            beats_per_section = 16  # 4小節 @ 4/4
            sections = []
            step = max(beats_per_section // 2, 1)
            for i in range(0, len(beats) - beats_per_section, step):
                t_start = float(beats[i])
                t_end   = float(beats[min(i + beats_per_section, len(beats) - 1)])
                s_start = int(t_start * sr)
                s_end   = int(min(t_end * sr, len(y)))
                if s_end - s_start < sr:
                    continue
                ss, sn = self._estimate_key(y[s_start:s_end], sr)
                sections.append((t_start, ss, sn))
            if not sections:
                ss, sn = self._estimate_key(y, sr)
                return [(0.0, ss, sn)]
            # 連続同キー区間を統合
            modulations = [sections[0]]
            for t, ss, sn in sections[1:]:
                if ss != modulations[-1][1]:
                    modulations.append((t, ss, sn))
            return modulations
        except Exception:
            ss, sn = self._estimate_key(y, sr)
            return [(0.0, ss, sn)]

    def _scale_set_at(self, modulations, t):
        """指定時刻に対応するスケールセットを返す。"""
        result = modulations[0][1] if modulations else None
        for mt, ss, _ in modulations:
            if mt <= t:
                result = ss
            else:
                break
        return result

    def _score_audio(self, original_y, synth_y, sr):
        """スペクトル差分による採譜品質スコア（0-100）。原音波形は評価のみに使用。"""
        try:
            n = min(len(original_y), len(synth_y))
            if n < sr:
                return 50.0
            orig = original_y[:n]
            synt = synth_y[:n]
            # MFCC ベースの類似度（構造的類似性）
            mfcc_orig = librosa.feature.mfcc(y=orig, sr=sr, n_mfcc=13, hop_length=HOP)
            mfcc_synt = librosa.feature.mfcc(y=synt, sr=sr, n_mfcc=13, hop_length=HOP)
            n_frames = min(mfcc_orig.shape[1], mfcc_synt.shape[1])
            if n_frames < 4:
                return 50.0
            diff = mfcc_orig[:, :n_frames] - mfcc_synt[:, :n_frames]
            mse = float(np.mean(diff ** 2))
            score = max(0.0, min(100.0, 100.0 - mse * 0.1))
            return score
        except Exception:
            return 50.0

    def _iterative_refine(self, notes, chords, tempo, scale_set, original_y, sr,
                          midi_path, parts_fn, drum_events, duration):
        """反復改善ループ: 生成→評価→パラメータ微調整を2回繰り返す。
        原音波形は評価スコア算出のみに使い、出力には混ぜない。"""
        best_notes = notes
        best_score = 0.0

        for iteration in range(2):
            self._log(f"  反復改善 #{iteration + 1}...", -1)
            # 現在のノートで合成して評価
            try:
                parts = parts_fn(best_notes)
                self._save_midi(parts, drum_events, tempo, midi_path)
                synth = self._synthesize_audio(midi_path, parts, drum_events, duration)
                if synth is not None:
                    s_mono = synth.mean(axis=1) if synth.ndim == 2 else synth
                    score = self._score_audio(original_y, s_mono, sr)
                    self._log(f"    スコア: {score:.1f}/100", -1)
                    if score > best_score:
                        best_score = score
                        best_notes = notes
                else:
                    score = 0.0
            except Exception:
                score = 0.0

            # 微調整: スケールスナップの強度を変える
            if scale_set and iteration == 0:
                refined = self._refine_notes(best_notes, scale_set, tempo)
                if len(refined) > len(best_notes) * 0.5:
                    notes = refined
            elif iteration == 1:
                # 2回目: コードフィルタの閾値を緩和
                notes = self._filter_notes_by_chord(best_notes, chords)

        return best_notes

    # ---- 高精度CQT多声部検出 v5 ----------------------------

    def _detect_all_notes(self, y_h, sr):
        """CQT + オンセット同期 + 適応スレッショルド + 強度比ベース倍音除去"""
        from scipy.ndimage import median_filter, uniform_filter1d

        cfg = self._get_transcription_cfg()
        dur_floor = cfg['dur_floor_s']
        thr_delta = cfg['cqt_thr_delta']
        thr_min = cfg['cqt_thr_min']

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
        onsets_fr = librosa.onset.onset_detect(
            y=y_h, sr=sr, hop_length=HOP,
            onset_envelope=onset_env,
            backtrack=True,
        )
        onset_set = set(onsets_fr.tolist())

        # 適応スレッショルド: 各フレームの上位N%をノートとみなす
        self._log("  ノートトラッキング中...", 33)
        active = {}
        events = []
        for fi in range(C_sm.shape[1]):
            frame = C_sm[:, fi]
            frame_max = np.max(frame)
            thr = max(frame_max - thr_delta, thr_min)

            on_now = set()
            peaks = []
            for bi in range(1, n_bins - 1):
                if (frame[bi] > thr and
                        frame[bi] >= frame[bi-1] and frame[bi] >= frame[bi+1]):
                    peaks.append((bi, frame[bi]))

            # 倍音除去（オクターブ+5度のみ: 12, 24 半音）:
            # コードの構成音を誤除去しないよう最小限に抑える
            if peaks:
                peaks.sort(key=lambda x: x[1], reverse=True)
                kept = []
                kept_energy = {}
                for bi, en in peaks:
                    mn = int(midi_notes[bi])
                    is_harmonic = False
                    for km, ke in kept_energy.items():
                        diff = mn - km
                        if diff in (12, 24) and en < ke * 0.4:
                            is_harmonic = True
                            break
                    if not is_harmonic:
                        kept.append((bi, en, mn))
                        kept_energy[mn] = en
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
                    if dur >= dur_floor:
                        vel = int(np.clip(mx * 3.5 + 30, 35, 127))
                        events.append((times[sf_], dur, mn, vel))

        for mn, (sf_, mx, has_onset) in active.items():
            dur = times[-1] - times[sf_]
            if dur >= dur_floor:
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
        # 振幅エンベロープ（RMS）を同じフレーム数で算出
        rms = librosa.feature.rms(y=y_h, frame_length=HOP * 2, hop_length=HOP)[0]
        return self._f0_to_notes(f0, voiced, times, 8, rms)

    # ---- pyin ベース検出 ----------------------------------

    def _detect_bass(self, y_h, sr):
        self._log("  ベースライン解析中...", 37)
        nyq = sr / 2
        b, a = butter(4, min(250 / nyq, 0.99), btype="low")
        yb = filtfilt(b, a, y_h)
        # ベース域拡張: C4 まで拾えるようにし、hop も高解像度に
        f0, voiced, _ = librosa.pyin(
            yb, fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C4"), sr=sr, hop_length=HOP)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
        rms = librosa.feature.rms(y=yb, frame_length=HOP * 2, hop_length=HOP)[0]
        return self._f0_to_notes(f0, voiced, times, 4, rms)

    def _f0_to_notes(self, f0, voiced, times, max_gap_hz, rms=None):
        """f0 系列をノート化。振幅エンベロープ rms があれば、その区間平均から velocity を算出。"""
        events = []
        # 振幅正規化用の参照値（全体の95%パーセンタイル）
        rms_ref = float(np.percentile(rms, 95)) if (rms is not None and len(rms) > 0) else 0.0
        if rms_ref <= 1e-6:
            rms_ref = 1.0
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
                if dur >= 0.04:
                    midi = int(np.clip(np.round(
                        librosa.hz_to_midi(np.mean(hz_list))), 0, 127))
                    # 振幅ベースの velocity（区間の最大振幅を基準値で正規化）
                    if rms is not None and len(rms) > 0:
                        seg = rms[i:min(j, len(rms))]
                        amp = float(np.max(seg)) if len(seg) > 0 else 0.0
                        amp_norm = min(amp / rms_ref, 1.5)  # 1.0 超過も許容
                        vel = int(np.clip(30 + np.sqrt(amp_norm) * 90, 35, 120))
                    else:
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
            'maj':  [1,0,0,0,1,0,0,1,0,0,0,0],
            'min':  [1,0,0,1,0,0,0,1,0,0,0,0],
            'dom7': [1,0,0,0,1,0,0,1,0,0,1,0],
            'min7': [1,0,0,1,0,0,0,1,0,0,1,0],
            'maj7': [1,0,0,0,1,0,0,1,0,0,0,1],
            'dim':  [1,0,0,1,0,0,1,0,0,0,0,0],
            'aug':  [1,0,0,0,1,0,0,0,1,0,0,0],
            'sus4': [1,0,0,0,0,1,0,1,0,0,0,0],
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
        parts = {name: [] for name in self._RAW_INST_NAMES}
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
            return self._process_ai(input_path, output_path)
        except Exception as e:
            import traceback
            self._log(f"エラー: {e}", -1)
            return False, traceback.format_exc()

    # ============================================================
    # v7.0: AI パイプライン (Demucs + Basic Pitch + FluidSynth)
    # ============================================================

    def _process_ai(self, input_path: str, output_path: str):
        """Demucs + Basic Pitch の主パイプライン（100% MIDI再合成・原盤不使用）。

        Demucs ステムは MIDI 採譜の入力と評価スコア算出にのみ使用し、
        最終出力 WAV/MP3 には一切混合しない（原盤権侵害の防止）。
        """
        self._log("AIモデル準備中...", 2)
        if not _ensure_ai_packages(lambda m: self._log(m, 3)):
            return False, "AIパッケージの取得に失敗しました。pip install demucs basic-pitch を実行してください。"

        # 1. 原音ロード
        y, sr = self._load(input_path)
        duration = len(y) / sr

        # 2. Demucs でステム分離
        self._log("Demucs でステム分離中 (初回はモデル~80MBをダウンロード)...", 8)
        stems = self._demucs_separate(input_path)
        if stems is None:
            return False, "Demucs でのステム分離に失敗しました。"

        # 3. 各ステムから採譜
        tempo, beats = self._tempo(y, sr)
        self._current_tempo = float(tempo) if tempo else 120.0
        self._log(f"テンポ: {tempo:.1f} BPM", 28)

        scale_set, scale_name = self._estimate_key(y, sr)
        if scale_set is not None:
            self._log(f"  推定キー: {scale_name}", 30)

        # 転調検出（Sound Horizon 対応）
        modulations = self._detect_modulation(y, sr, beats)
        if len(modulations) > 1:
            self._log(f"  転調検出: {len(modulations)} キー変化", 30)

        tq = getattr(self, 'transcription_quality', 'balanced')
        self._log(f"採譜モード: {tq}", 30)

        # --- ステージ1: コード解析（後処理で使用） ---
        self._log("コード進行解析中...", 31)
        y_h, _ = librosa.effects.hpss(y, margin=2.0)
        chords = self._detect_chords(y_h, sr, beats)
        self._log(f"  コード: {len(chords)} 区間", 32)

        # --- ステージ2: 各ステム採譜 (Basic Pitch / pyin) ---
        skip_vocal = (self.mode == "ai_inst")
        if skip_vocal:
            self._log("ガイドメロディなしモード: ボーカル採譜をスキップ", 33)
            vocal_notes = []
        else:
            self._log("ボーカルを多声部採譜中 (Basic Pitch)...", 33)
            vocal_stem = self._preprocess_stem(stems["vocals"], sr, 'vocals')
            vocal_notes = self._basic_pitch_notes(vocal_stem, sr, is_vocal=True)
            vocal_notes = self._transformer_refine(vocal_notes)
            vocal_notes = self._remove_overlapping_notes(vocal_notes)
            self._log_midi_stats("ボーカル採譜結果", vocal_notes)

        self._log("その他楽器を多声部採譜中 (Basic Pitch)...", 42)
        other_stem = self._preprocess_stem(stems["other"], sr, 'other')
        other_notes = self._basic_pitch_notes(other_stem, sr, is_vocal=False)
        other_notes = self._transformer_refine(other_notes)
        other_notes = self._remove_overlapping_notes(other_notes)
        self._log_midi_stats("その他採譜結果", other_notes)

        self._log("ベースライン採譜中 (pyin)...", 50)
        bass_stem = self._preprocess_stem(stems["bass"], sr, 'bass')
        bass_notes = self._pyin_bass_notes(bass_stem, sr)
        self._log_midi_stats("ベース採譜結果", bass_notes)

        self._log("ドラム採譜中...", 56)
        drum_events = self._drums_from_stem(stems["drums"], sr)
        self._log(f"  ドラム: {len(drum_events)} イベント", 58)

        # --- ステージ3: ノート後処理パイプライン ---
        self._log("ノート後処理中...", 60)

        # 3a. _refine_notes: スケール補正・極短削除・結合・跳躍抑制
        self._log("  ノート精製 (refine)...", 62)
        vocal_notes = self._refine_notes(vocal_notes, scale_set, tempo)
        other_notes = self._refine_notes(other_notes, scale_set, tempo)
        bass_notes = self._refine_notes(bass_notes, None, tempo)  # ベースはスケール補正なし

        # 3b. _filter_notes_by_chord: コード制約フィルタ
        self._log("  コード制約フィルタ...", 64)
        other_notes = self._filter_notes_by_chord(other_notes, chords)

        # 3c. _rebuild_rhythm: 16分グリッドへの軽量スナップ
        self._log("  リズム再構築...", 66)
        vocal_notes = self._rebuild_rhythm(vocal_notes, tempo)
        other_notes = self._rebuild_rhythm(other_notes, tempo)
        bass_notes = self._rebuild_rhythm(bass_notes, tempo)

        # 3d. _fill_harmony: スカスカ区間にコードトーンを補完
        self._log("  ハーモニー補完...", 68)
        other_notes = self._fill_harmony(other_notes, chords, tempo)

        # 3e. Sound Horizon 最適化: コード内声部充填 + サスティン延長
        self._log("  Sound Horizon 最適化...", 69)
        other_notes = self._thicken_chords(other_notes, chords)
        other_notes = self._extend_sustains(other_notes)
        vocal_notes = self._extend_sustains(vocal_notes)

        # 3f. 音楽理論補正 (AdvancedMusicTheoryCorrector: ①〜⑨)
        try:
            from music_theory import (
                AdvancedMusicTheoryCorrector, DelayedStabilizer,
                smooth_chord_transitions,
            )
            self._log("  音楽理論補正中 (9機能)...", 70)
            mtc = AdvancedMusicTheoryCorrector(
                audio=y, sr=sr,
                tempo=self._current_tempo,
                beat_times=beats,
                chord_window=8,
                section_sec=16.0,
            )
            # ① 遅延許容型安定化 + ③ コード遷移制御
            if mtc.chords:
                stab = DelayedStabilizer(look_ahead=3, look_back=3)
                mtc.chords = smooth_chord_transitions(
                    stab.stabilize_chords(mtc.chords), hysteresis=0.10
                )
            if mtc.key_score > 0.3:
                self._log(
                    f"  推定キー: {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][mtc.key_root]} {mtc.key_mode} "
                    f"(score={mtc.key_score:.2f}) "
                    f"sections={len(mtc._section_key.sections)}", 70)
            vocal_notes = mtc.correct(
                vocal_notes, audio=y,
                apply_bass=False,
                apply_bass_opt=False,
                apply_triad=False,       # メロディはトライアド強制しない
                max_polyphony=2,
                quantize_strength=0.4,
                top_k_per_window=2,
            )
            other_notes = mtc.correct(
                other_notes, audio=y,
                apply_bass=False,
                apply_triad=True,
                apply_bass_opt=False,
                max_polyphony=5,
                quantize_strength=0.65,
                top_k_per_window=5,
            )
            bass_notes = mtc.correct(
                bass_notes, audio=y,
                apply_scale=False,
                apply_chord=False,
                apply_harmony=False,
                apply_triad=False,
                apply_bass=True,
                apply_bass_opt=True,
                quantize_strength=0.7,
                top_k_per_window=2,
            )
            self._log(
                f"  理論補正完了: vocal={len(vocal_notes)} "
                f"other={len(other_notes)} bass={len(bass_notes)}", 71)
        except ImportError:
            pass  # music_theory.py が無くても既存動作に影響なし
        except Exception as e:
            self._log(f"  音楽理論補正スキップ: {e}", 71)

        # 3g. ヒューマナイゼーション (人間演奏感の付与)
        try:
            from music_theory import humanize_notes
            self._log("  ヒューマナイズ中...", 71)
            # メロディ: ジッター強め、最小持続短め (歌は持続)
            vocal_notes = humanize_notes(
                vocal_notes, audio=y, sr=sr,
                tempo=self._current_tempo, beat_times=beats,
                min_dur=0.10, jitter_sec=0.012, vel_jitter=5,
                max_polyphony=2, max_bass_jump=12,
                accent_boost=8, seed=42,
            )
            # 伴奏: 中庸
            other_notes = humanize_notes(
                other_notes, audio=y, sr=sr,
                tempo=self._current_tempo, beat_times=beats,
                min_dur=0.08, jitter_sec=0.010, vel_jitter=4,
                max_polyphony=5, max_bass_jump=12,
                accent_boost=10, seed=43,
            )
            # ベース: ジッター控えめ、ジャンプ厳格
            bass_notes = humanize_notes(
                bass_notes, audio=y, sr=sr,
                tempo=self._current_tempo, beat_times=beats,
                min_dur=0.10, jitter_sec=0.006, vel_jitter=3,
                max_polyphony=1, max_bass_jump=5,
                midi_range=(24, 60), accent_boost=12, seed=44,
            )
        except ImportError:
            pass
        except Exception as e:
            self._log(f"  ヒューマナイズスキップ: {e}", 71)

        # 3g. Velocity はクランプのみ（元のダイナミクスを潰さない）
        vocal_notes = self._normalize_velocity(vocal_notes, 50, 120)
        other_notes = self._normalize_velocity(other_notes, 35, 115)
        bass_notes  = self._normalize_velocity(bass_notes,  55, 120)

        self._log_midi_stats("最終ボーカル", vocal_notes)
        self._log_midi_stats("最終その他", other_notes)
        self._log_midi_stats("最終ベース", bass_notes)

        # --- ステージ4: 楽器割り当て ---
        self._log("パート分離・楽器割り当て中...", 72)
        parts = self._ai_assign_parts(vocal_notes, other_notes, bass_notes)
        parts = self._consolidate_parts(parts)
        parts = self._generate_pad_from_chords(parts, tempo, duration)
        active = {k: len(v) for k, v in parts.items() if v}
        self._log(f"  パート構成: {active}", 73)

        # --- ステージ4.5: 生産品質ヒューマナイズ ---
        self._log("生産品質ヒューマナイズ中...", 73)
        parts, drum_events = self._production_humanize(
            parts, drum_events, tempo, beats, seed=42
        )

        # --- ステージ5: MIDI 保存 ---
        self._log("MIDIを保存中...", 75)
        midi_path = str(Path(output_path).with_suffix(".mid"))
        self._save_midi(parts, drum_events, tempo, midi_path)

        # --- ステージ6: 反復改善ループ (評価→微調整) ---
        self._log("反復改善ループ実行中...", 78)
        try:
            all_notes = vocal_notes + other_notes + bass_notes
            def _make_parts(ns):
                # 改善後ノートを vocal/other/bass に再分割して assign
                return self._ai_assign_parts(
                    [n for n in ns if n in vocal_notes] or vocal_notes,
                    [n for n in ns if n in other_notes] or other_notes,
                    [n for n in ns if n in bass_notes] or bass_notes,
                )
            _ = self._iterative_refine(
                all_notes, chords, tempo, scale_set, y, sr,
                midi_path, lambda ns: parts, drum_events, duration
            )
        except Exception:
            pass  # 反復失敗時は元のパートを使う

        # --- ステージ7: 合成 (MIDIベースのみ、原音は一切混ぜない) ---
        self._log("音声合成中...", 82)
        synth_audio = self._synthesize_audio(midi_path, parts, drum_events, duration)

        # 原盤権保護: 合成音のみ使用（Demucs ステムは最終出力に一切含めない）
        self._log("マスタリング中...", 88)
        audio = synth_audio.copy()

        # 採譜・評価が完了したので Demucs 由来の原音ステムを破棄する。
        # メモリ上の派生物をユーザー環境に残さないことで原盤権侵害を防ぐ。
        try:
            stems.clear()
        except Exception:
            pass
        del stems

        # --- ステージ8: マスタリング ---
        audio = self._master(audio)

        self._log("出力品質チェック中...", 91)
        self._validate_output(audio, duration)

        # 周波数バランスレポート
        try:
            self._freq_balance_report(audio, y, output_path)
        except Exception:
            pass

        # --- ステージ9: 保存 ---
        self._log("MP3を保存中...", 94)
        self._save_mp3(audio, output_path)

        # LUFS 測定
        lufs = self._measure_lufs(audio)
        self._log(f"出力ラウドネス: {lufs:.1f} LUFS (目標: -14 LUFS)", 97)

        # ライセンスファイル自動生成
        try:
            _generate_license_readme()
        except Exception:
            pass

        self._log("完了！", 100)
        return True, output_path

    # ---- Demucs ステム分離 --------------------------------

    def _demucs_separate(self, input_path):
        """Demucs (htdemucs) でステム分離し、dict{name: ndarray} を返す。

        ⚠ 原盤権ポリシー: 返却される ndarray は MIDI 採譜の入力と
        評価スコア算出にのみ使用すること。最終 WAV/MP3 への混入は禁止。
        ステムはディスクに保存せずメモリ内のみで保持し、_process_ai 終了時に
        破棄される。
        """
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

    # 旧 _mix_stems (Demucs 出力をミックスする関数) は v7.x で削除。
    # 原盤権侵害防止のため、ステムを混合するヘルパは提供しない。

    # ---- Basic Pitch 多声部採譜 ---------------------------

    def _basic_pitch_notes(self, audio, sr, is_vocal=False):
        """Basic Pitch で多声部採譜し [(t, dur, midi, vel), ...] を返す"""
        # Windows で非実在ディレクトリが PATH や DLL 検索パスに含まれていると
        # Basic Pitch / pyfluidsynth 等の依存が import 時に WinError 3 を投げる。
        # 対策: (1) PATH サニタイズ  (2) os.add_dll_directory のガード
        original_path = os.environ.get("PATH", "")
        sep = os.pathsep
        _patched_add_dll = False
        _orig_add_dll = None
        try:
            clean = [p for p in original_path.split(sep)
                     if not p or Path(p).exists()]
            os.environ["PATH"] = sep.join(clean)
        except Exception:
            pass
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            _orig_add_dll = os.add_dll_directory
            def _safe_add_dll(path):
                try:
                    if Path(path).is_dir():
                        return _orig_add_dll(path)
                except OSError:
                    pass
                return None
            os.add_dll_directory = _safe_add_dll
            _patched_add_dll = True

        try:
            try:
                from basic_pitch.inference import predict
                from basic_pitch import ICASSP_2022_MODEL_PATH
            except Exception as e:
                err_type = type(e).__name__
                msg = str(e)[:150]
                self._log(f"Basic Pitch 利用不可 ({err_type}): {msg}", -1)
                self._log("  → CQT+pyin フォールバックで代替（精度は低下します）", -1)
                return self._fallback_transcribe(audio, sr, is_vocal)
        finally:
            os.environ["PATH"] = original_path
            if _patched_add_dll and _orig_add_dll is not None:
                os.add_dll_directory = _orig_add_dll

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
            cfg = self._get_transcription_cfg()
            try:
                _, _, note_events = predict(
                    tmp,
                    model_or_model_path=str(ICASSP_2022_MODEL_PATH),
                    onset_threshold=(
                        cfg['onset_threshold_vocal'] if is_vocal
                        else cfg['onset_threshold']
                    ),
                    frame_threshold=(
                        cfg['frame_threshold_vocal'] if is_vocal
                        else cfg['frame_threshold']
                    ),
                    minimum_note_length=(
                        cfg['min_note_ms_vocal'] if is_vocal
                        else cfg['min_note_ms']
                    ),
                    minimum_frequency=(
                        cfg['freq_min_vocal'] if is_vocal
                        else cfg['freq_min']
                    ),
                    maximum_frequency=(
                        cfg['freq_max_vocal'] if is_vocal
                        else cfg['freq_max']
                    ),
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
            start = float(ev[0])
            end = float(ev[1])
            midi = int(ev[2])
            amp = float(ev[3]) if len(ev) > 3 else 0.8
            dur = end - start
            if dur < 0.02:
                continue  # 0.02秒未満のみ削除（削りすぎ防止）
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
        """Demucs ベースステムを高精度pyin採譜 + オクターブ補正"""
        if np.max(np.abs(audio)) < 1e-4:
            return []
        # E1 (41Hz) から C4 まで、高解像度 hop で検出
        f0, voiced, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("E1"),
            fmax=librosa.note_to_hz("C4"), sr=sr, hop_length=HOP
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
        rms = librosa.feature.rms(y=audio, frame_length=HOP * 2, hop_length=HOP)[0]
        notes = self._f0_to_notes(f0, voiced, times, 4, rms)
        # オクターブ補正: ベースは通常 E1-E3 (midi 28-52) の範囲
        corrected = []
        for (t, dur, midi, vel) in notes:
            while midi > 52:
                midi -= 12
            while midi < 28 and midi + 12 <= 52:
                midi += 12
            corrected.append((t, dur, midi, vel))
        return corrected

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
            mid_lo = fft[(freqs >= 120) & (freqs < 200)].sum() / tot
            mid = fft[(freqs >= 200) & (freqs < 1500)].sum() / tot
            hi = fft[freqs >= 3000].sum() / tot
            if lo > 0.45:
                kind = "kick"
            elif hi > 0.55:
                kind = "hihat"
            elif hi > 0.35 and mid < 0.2:
                kind = "open_hat"
            elif mid > 0.3 and hi > 0.15:
                kind = "ride"
            elif mid_lo > 0.25 and lo < 0.35:
                kind = "tom"
            else:
                kind = "snare"
            peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
            vel = int(np.clip(60 + peak * 120, 40, 127))
            events.append((ot, kind, vel))
        return events

    # ---- AI 検出結果 → 15楽器割り当て ----------------------

    def _ai_assign_parts(self, vocal_notes, other_notes, bass_notes):
        """15楽器フルアレンジ: メロディ・伴奏・ベースを豊かな音色で彩る"""
        parts = {name: [] for name in self._RAW_INST_NAMES}

        # ステムごとの相対エネルギーからダブリング倍率を動的に算出
        def _stem_energy(notes, ref=0.7):
            if not notes:
                return ref
            vels = np.array([v for _, _, _, v in notes], dtype=np.float32)
            return float(np.clip(np.percentile(vels, 75) / 100.0, 0.2, 1.4))

        v_energy = _stem_energy(vocal_notes)
        inv_v = 1.0 / max(v_energy, 0.5)
        flute_scale = float(np.clip(0.35 * inv_v, 0.15, 0.55))
        violin_scale = float(np.clip(0.35 * inv_v, 0.15, 0.45))
        epiano_scale = float(np.clip(0.30 * inv_v, 0.12, 0.40))

        # === 1. メロディ (ボーカル採譜) → ピアノ主体 + ダブリング ===
        for (t, dur, midi, vel) in vocal_notes:
            parts['piano'].append((t, dur, midi, vel))
            if midi >= 80:
                parts['flute'].append((t, dur, midi, int(vel * flute_scale)))
            elif midi >= 68:
                parts['violin'].append((t, dur, midi, int(vel * violin_scale)))
            else:
                parts['e_piano'].append((t, dur, midi, int(vel * epiano_scale)))
            if dur >= 0.8 and 60 <= midi <= 84:
                parts['trumpet'].append((t, min(dur, 0.5), midi, int(vel * 0.25)))

        # === 2. 伴奏 (other ステム) → 音域・持続時間で多楽器に振り分け ===
        other_sorted = sorted(other_notes, key=lambda x: x[0])

        for (t, dur, midi, vel) in other_sorted:
            if midi >= 84:
                parts['glockenspiel'].append((t, dur, midi, int(vel * 0.8)))
                if dur >= 0.3:
                    parts['choir'].append((t, dur, midi, int(vel * 0.25)))
            elif midi >= 72:
                parts['e_piano'].append((t, dur, midi, vel))
                if dur >= 0.5:
                    parts['violin'].append((t, dur, midi, int(vel * 0.3)))
            elif midi >= 60:
                if dur >= 0.5:
                    parts['strings'].append((t, dur, midi, vel))
                    parts['organ'].append((t, dur, midi, int(vel * 0.25)))
                else:
                    parts['guitar_clean'].append((t, dur, midi, vel))
            elif midi >= 48:
                parts['guitar_nylon'].append((t, dur, midi, vel))
                if dur >= 0.5:
                    parts['viola'].append((t, dur, midi, int(vel * 0.35)))
            else:
                parts['cello'].append((t, dur, midi, vel))

        # === 3. ベース → ベース + チェロオクターブダブリング ===
        for (t, dur, midi, vel) in bass_notes:
            parts['bass'].append((t, dur, midi, vel))
            if midi + 12 < 60:
                parts['cello'].append((t, dur, midi + 12, int(vel * 0.40)))

        # === 4. 持続音にパッド/コーラス補強 ===
        long_strings = [(t, dur, m, v) for t, dur, m, v in parts['strings'] if dur >= 1.0]
        for (t, dur, midi, vel) in long_strings:
            parts['pad'].append((t, dur, midi, int(vel * 0.35)))
            parts['choir'].append((t, dur, midi, int(vel * 0.20)))

        # === 5. ポリフォニー制限（位相干渉・音の濁り防止） ===
        MAX_POLY = {'piano': 6, 'e_piano': 4, 'guitar_clean': 4,
                    'guitar_nylon': 4, 'strings': 6, 'glockenspiel': 3,
                    'cello': 3, 'bass': 2, 'pad': 4, 'choir': 4}
        for name in parts:
            limit = MAX_POLY.get(name, 4)
            evts = sorted(parts[name], key=lambda x: x[0])
            filtered = []
            for note in evts:
                t, dur, _, vel = note
                # 現在アクティブなノートのインデックスと velocity を集める
                active_idx = [i for i, (ft, fd, _, _) in enumerate(filtered)
                              if ft + fd > t and ft <= t]
                if len(active_idx) < limit:
                    filtered.append(note)
                else:
                    # 最小 velocity のアクティブノートを見つけ、今のノートの方が大きければ置換
                    min_i = min(active_idx, key=lambda i: filtered[i][3])
                    if vel > filtered[min_i][3]:
                        # 既存の最弱ノートを切る（現時刻まで短縮）→ 新ノートを追加
                        ft, fd, fm, fv = filtered[min_i]
                        new_dur = max(0.01, t - ft)
                        filtered[min_i] = (ft, new_dur, fm, fv)
                        filtered.append(note)
            parts[name] = filtered

        return parts

    # ---- パート統合 (15楽器 → 最大6パート) ----------------

    def _consolidate_parts(self, parts):
        """15楽器パートを音域・同時発音で melody/chord/bass/pad/decoration に分離。

        全ノートをプールし、音楽的役割で再分配する:
          bass:       MIDI 36-55 (E2-G3)
          chord:      同時3音以上クラスタ (MIDI 48-72)
          melody:     最高音・単音 (MIDI 60-96)
          pad:        コード進行から自動生成される持続音 (後段で追加)
          decoration: 高音域装飾音 (MIDI 84+)
          sub_melody: 残りの中音域
        """
        all_notes = []
        for name, evts in parts.items():
            all_notes.extend(evts)
        if not all_notes:
            return {name: [] for name in self.MIDI_MAP}

        all_notes.sort(key=lambda x: x[0])

        # --- 同時発音クラスタ検出 (±50ms) ---
        clusters = []
        current_cluster = [all_notes[0]]
        for note in all_notes[1:]:
            if abs(note[0] - current_cluster[0][0]) <= 0.05:
                current_cluster.append(note)
            else:
                clusters.append(current_cluster)
                current_cluster = [note]
        clusters.append(current_cluster)

        bass_notes = []
        chord_notes = []
        melody_notes = []
        decoration_notes = []
        assigned = set()

        for cluster in clusters:
            if len(cluster) >= 3:
                mid_range = [n for n in cluster if 48 <= n[2] <= 72]
                if len(mid_range) >= 3:
                    for n in mid_range:
                        chord_notes.append(n)
                        assigned.add(id(n))
                    for n in cluster:
                        if id(n) not in assigned:
                            if n[2] > 72:
                                melody_notes.append(n)
                                assigned.add(id(n))
                    continue

            for n in cluster:
                if id(n) in assigned:
                    continue
                if n[2] <= 55:
                    bass_notes.append(n)
                    assigned.add(id(n))

        for cluster in clusters:
            unassigned = [n for n in cluster if id(n) not in assigned]
            if not unassigned:
                continue
            if len(unassigned) == 1:
                n = unassigned[0]
                if 60 <= n[2] <= 96:
                    melody_notes.append(n)
                elif n[2] > 96:
                    decoration_notes.append(n)
                elif n[2] >= 48:
                    chord_notes.append(n)
                else:
                    bass_notes.append(n)
                assigned.add(id(n))
            else:
                highest = max(unassigned, key=lambda x: x[2])
                if highest[2] >= 60:
                    melody_notes.append(highest)
                    assigned.add(id(highest))
                for n in unassigned:
                    if id(n) not in assigned:
                        if n[2] >= 84:
                            decoration_notes.append(n)
                        elif 48 <= n[2] <= 72:
                            chord_notes.append(n)
                        elif n[2] < 48:
                            bass_notes.append(n)
                        else:
                            melody_notes.append(n)
                        assigned.add(id(n))

        for n in all_notes:
            if id(n) not in assigned:
                if n[2] <= 55:
                    bass_notes.append(n)
                elif n[2] >= 84:
                    decoration_notes.append(n)
                elif n[2] >= 60:
                    melody_notes.append(n)
                else:
                    chord_notes.append(n)

        def _drop_weak(notes, min_count=3):
            if len(notes) < min_count:
                return []
            vels = [v for _, _, _, v in notes]
            if np.mean(vels) < 25:
                return []
            return notes

        consolidated = {
            'melody': _drop_weak(melody_notes),
            'chord': _drop_weak(chord_notes),
            'bass': _drop_weak(bass_notes),
            'decoration': _drop_weak(decoration_notes, min_count=5),
            'sub_melody': [],
        }

        # melody は最重要 — ドロップされた場合でも元データがあれば復活
        if not consolidated['melody'] and melody_notes:
            consolidated['melody'] = melody_notes

        return consolidated

    def _generate_pad_from_chords(self, parts, tempo, duration):
        """chord パートからルート音を抽出し、持続音パッドを自動生成する。

        各小節の先頭コードからルート+5度+オクターブ上を全音符で生成。
        ボリュームは他パートの -12dB (= 0.25倍) に設定。
        """
        chord_notes = parts.get('chord', [])
        if not chord_notes:
            return parts

        config = _load_instrument_config()
        if config and not config.get("pad_layer", True):
            return parts

        beat_dur = 60.0 / max(tempo, 60)
        bar_dur = beat_dur * 4
        n_bars = int(duration / bar_dur) + 1

        chord_sorted = sorted(chord_notes, key=lambda x: x[0])
        pad_notes = []

        for bar_idx in range(n_bars):
            bar_start = bar_idx * bar_dur
            bar_end = bar_start + bar_dur
            bar_chords = [n for n in chord_sorted
                          if bar_start - 0.1 <= n[0] < bar_end]
            if not bar_chords:
                continue
            root_midi = min(n[2] for n in bar_chords)
            avg_vel = int(np.mean([n[3] for n in bar_chords]) * 0.25)
            avg_vel = max(20, min(70, avg_vel))

            pad_dur = min(bar_dur, duration - bar_start)
            if pad_dur < 0.5:
                continue

            root = max(48, min(60, root_midi))
            pad_notes.append((bar_start, pad_dur, root, avg_vel))
            pad_notes.append((bar_start, pad_dur, root + 7, avg_vel))
            pad_notes.append((bar_start, pad_dur, root + 12, int(avg_vel * 0.8)))

        parts['pad'] = pad_notes
        return parts

    # ---- ノート量子化・正規化 ----------------------------

    def _quantize_notes(self, notes, beat_times, tempo):
        """極めて軽い量子化: 50ms以内のみ補正、強制吸着なし。"""
        if not notes or beat_times is None or len(beat_times) < 2:
            return notes

        beat_dur = 60.0 / max(tempo, 40)
        grid_step = beat_dur / 4  # 16分
        max_t = beat_times[-1] + beat_dur * 4
        grid = np.arange(0, max_t, grid_step)
        snap_limit = 0.05  # 50ms

        quantized = []
        for (t, dur, midi, vel) in notes:
            idx = np.argmin(np.abs(grid - t))
            qt = grid[idx]
            dist = abs(qt - t)
            # 50ms以内のみ 40% スナップ（強制吸着なし）
            if dist <= snap_limit:
                qt = t + (qt - t) * 0.4
            else:
                qt = t
            # duration はそのまま保持（崩さない）
            quantized.append((qt, dur, midi, vel))

        # 同一時刻・同一ピッチの重複除去（loudest 優先）
        seen = {}
        for (t, dur, midi, vel) in quantized:
            key = (round(t, 3), midi)
            if key in seen:
                ot, od, om, ov = seen[key]
                if vel > ov:
                    seen[key] = (t, dur, midi, vel)
            else:
                seen[key] = (t, dur, midi, vel)
        return sorted(seen.values(), key=lambda x: x[0])

    def _normalize_velocity(self, notes, target_min=45, target_max=115):
        """Velocity はクランプのみ。元のダイナミクスを潰さない。"""
        if not notes:
            return notes
        return [(t, dur, m, int(np.clip(v, target_min, target_max)))
                for (t, dur, m, v) in notes]

    def _estimate_key(self, audio, sr):
        """Krumhansl–Schmuckler キー推定。
        戻り値: (scale_set: set[int 0-11], scale_name: str)
        失敗時は (None, "unknown")。
        """
        try:
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP)
            chroma_mean = np.mean(chroma, axis=1)
            if np.sum(chroma_mean) < 1e-6:
                return None, "unknown"
            major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            pitch_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            best_score = -1e9
            best_root = 0
            best_mode = 'major'
            for root in range(12):
                for mode_name, prof in (('major', major), ('minor', minor)):
                    rotated = np.roll(prof, root)
                    score = float(np.corrcoef(chroma_mean, rotated)[0, 1])
                    if score > best_score:
                        best_score = score
                        best_root = root
                        best_mode = mode_name
            if best_mode == 'major':
                degrees = [0, 2, 4, 5, 7, 9, 11]
            else:
                degrees = [0, 2, 3, 5, 7, 8, 10]
            scale_set = {(best_root + d) % 12 for d in degrees}
            scale_name = f"{pitch_names[best_root]} {best_mode}"
            return scale_set, scale_name
        except Exception:
            return None, "unknown"

    def _scale_snap_notes(self, notes, scale_set):
        """スケール外ノートを最寄りスケール音へスナップ（±2半音以内を検索）。
        短いノート（経過音・装飾音の可能性）はスナップせず原音の個性を保持する。"""
        if not notes or not scale_set:
            return notes
        snapped = []
        for (t, dur, midi, vel) in notes:
            pc = int(midi) % 12
            if dur < 0.35 or pc in scale_set:
                snapped.append((t, dur, midi, vel))
                continue
            best_delta = None
            for delta in [-1, 1, -2, 2]:
                if (pc + delta) % 12 in scale_set:
                    best_delta = delta
                    break
            if best_delta is not None:
                snapped.append((t, dur, midi + best_delta, vel))
            else:
                snapped.append((t, dur, midi, vel))
        return snapped

    # ---- ベロシティ & タイミング ヒューマナイズ (生産品質) ----

    # 楽器グループ別タイミングジッター (秒)
    _TIMING_JITTER = {
        'drums': 0.003,
        'melody': 0.010,
        'chord': 0.012,
        'bass': 0.005,
        'pad': 0.015,
        'decoration': 0.008,
        'sub_melody': 0.012,
    }

    def _production_humanize(self, parts, drum_events, tempo, beats, seed=42):
        """生産品質のヒューマナイズ:
        - ±5 ランダムベロシティ揺らぎ (seed=曲ハッシュで再現可能)
        - 拍頭(beat 1,3) +6、裏拍 -3 のアクセント
        - ベロシティ標準偏差<10のパートは破棄 (死んだ採譜)
        - フレーズ末尾 (1秒以上の無音直前3音) -8/-15/-22 減衰
        """
        rng = np.random.RandomState(seed)
        beat_times = np.array(beats) if beats is not None and len(beats) > 0 else np.array([])
        beat_interval = 60.0 / max(tempo, 60.0) if tempo else 0.5

        def _is_downbeat(t):
            if len(beat_times) == 0:
                return (t % (beat_interval * 2)) < beat_interval * 0.15
            diffs = np.abs(beat_times - t)
            if len(diffs) == 0:
                return False
            min_idx = int(np.argmin(diffs))
            return float(diffs[min_idx]) < 0.05 and min_idx % 2 == 0

        def _is_offbeat(t):
            if len(beat_times) == 0:
                phase = (t % beat_interval) / beat_interval
                return 0.35 < phase < 0.65
            closest_idx = int(np.argmin(np.abs(beat_times - t))) if len(beat_times) > 0 else 0
            if closest_idx < len(beat_times):
                offset = abs(t - beat_times[closest_idx])
                return offset > beat_interval * 0.35
            return False

        def _find_phrase_ends(notes):
            """1秒以上の無音区間の直前3音のインデックスを返す"""
            ends = set()
            sorted_notes = sorted(notes, key=lambda x: x[0])
            for i in range(len(sorted_notes) - 1):
                t_end = sorted_notes[i][0] + sorted_notes[i][1]
                t_next = sorted_notes[i + 1][0]
                if t_next - t_end >= 1.0:
                    for j in range(max(0, i - 2), i + 1):
                        ends.add(j)
            for j in range(max(0, len(sorted_notes) - 3), len(sorted_notes)):
                ends.add(j)
            return ends

        new_parts = {}
        for name, notes in parts.items():
            if not notes:
                new_parts[name] = []
                continue
            vels = [v for _, _, _, v in notes]
            if len(vels) > 1 and float(np.std(vels)) < 10:
                new_parts[name] = []
                continue
            jitter_s = self._TIMING_JITTER.get(name, 0.010)
            sorted_notes = sorted(notes, key=lambda x: x[0])
            phrase_ends = _find_phrase_ends(sorted_notes)
            humanized = []
            for i, (t, dur, midi, vel) in enumerate(sorted_notes):
                dt = rng.uniform(-jitter_s, jitter_s)
                new_t = max(0.0, t + dt)
                if _is_downbeat(t):
                    vel = min(127, vel + 6)
                elif _is_offbeat(t):
                    vel = max(20, vel - 3)
                vel = int(np.clip(vel + rng.randint(-5, 6), 20, 127))
                if i in phrase_ends:
                    dist_from_end = max(0, max(phrase_ends) - i) if phrase_ends else 0
                    decay_vals = [-8, -15, -22]
                    decay_idx = min(dist_from_end, 2)
                    vel = max(20, vel + decay_vals[decay_idx])
                humanized.append((new_t, dur, midi, vel))
            new_parts[name] = humanized

        new_drums = []
        for evt in drum_events:
            onset, kind = evt[0], evt[1]
            vel = evt[2] if len(evt) > 2 else 100
            dt = rng.uniform(-0.003, 0.003)
            onset = max(0.0, onset + dt)
            if _is_downbeat(evt[0]):
                vel = min(127, vel + 6)
            elif _is_offbeat(evt[0]):
                vel = max(25, vel - 3)
            vel = int(np.clip(vel + rng.randint(-5, 6), 25, 127))
            new_drums.append((onset, kind, vel))

        return new_parts, new_drums

    # ---- マルチSoundFont 合成 ----------------------------

    def _split_midi_by_channels(self, midi_path, channels, output_path):
        """MIDIファイルから指定チャンネルのみ抽出して新ファイルに保存"""
        try:
            import mido
        except ImportError:
            return False
        try:
            mid = mido.MidiFile(midi_path)
            out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
            ch_set = set(channels)
            for track in mid.tracks:
                new_track = mido.MidiTrack()
                for msg in track:
                    if msg.is_meta:
                        new_track.append(msg.copy())
                    elif hasattr(msg, "channel") and msg.channel in ch_set:
                        new_track.append(msg.copy())
                    elif hasattr(msg, "channel"):
                        new_track.append(mido.Message(
                            "note_off", channel=msg.channel,
                            note=0, velocity=0, time=msg.time
                        ))
                    else:
                        new_track.append(msg.copy())
                out.tracks.append(new_track)
            out.save(output_path)
            return True
        except Exception:
            return False

    def _render_with_sf2(self, midi_path, sf2_path, duration):
        """指定SF2でMIDIをFluidSynthレンダリングし、ndarray を返す"""
        fs_bin = _ensure_fluidsynth_cli(lambda m: None)
        if fs_bin is None:
            return None
        work_dir = Path(tempfile.mkdtemp(prefix="earcopy_msf2_"))
        ascii_midi = work_dir / "in.mid"
        ascii_wav = work_dir / "out.wav"
        ascii_sf2 = work_dir / "font.sf2"
        try:
            shutil.copy(midi_path, ascii_midi)
            try:
                ascii_sf2.symlink_to(sf2_path)
            except (OSError, NotImplementedError):
                shutil.copy(sf2_path, ascii_sf2)
            timeout_sec = max(300, int(duration * 8))
            cmd = [
                fs_bin, "-ni", "-F", str(ascii_wav),
                "-r", str(SR), "-g", "0.85",
                "-R", "0", "-C", "0", "-T", "wav",
                str(ascii_sf2), str(ascii_midi),
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    encoding="utf-8", errors="replace")
            proc.communicate(timeout=timeout_sec)
            if not ascii_wav.exists() or ascii_wav.stat().st_size < 1000:
                return None
            audio, _ = librosa.load(str(ascii_wav), sr=SR, mono=False)
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio]).astype(np.float32)
            elif audio.ndim == 2:
                audio = audio.T.astype(np.float32)
            target_len = int((duration + 0.5) * SR)
            if len(audio) > target_len:
                audio = audio[:target_len]
            return audio
        except Exception:
            return None
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _synthesize_multi_sf2(self, midi_path, duration):
        """楽器グループごとに最適なSF2でレンダリングし、パートEQ・空間処理付きでミックス"""
        config = _load_instrument_config()
        if config is None:
            return None
        groups = config.get("instrument_groups", {})

        general_sf2 = _find_sf2()
        if general_sf2 is None:
            return None

        piano_sf2 = _find_piano_sf2()
        rendered_groups = {}
        work_dir = Path(tempfile.mkdtemp(prefix="earcopy_multi_"))

        try:
            for gname, gcfg in groups.items():
                channels = gcfg.get("channels", [])
                if not channels:
                    continue
                sf2_key = gcfg.get("sf2_key", "general")
                if sf2_key == "salamander_piano" and piano_sf2:
                    sf2 = piano_sf2
                else:
                    sf2 = general_sf2

                group_midi = str(work_dir / f"{gname}.mid")
                if not self._split_midi_by_channels(midi_path, channels, group_midi):
                    continue
                audio = self._render_with_sf2(group_midi, sf2, duration)
                if audio is not None and np.max(np.abs(audio)) > 1e-6:
                    rendered_groups[gname] = (audio, gcfg)

            if not rendered_groups:
                return None

            self._log(f"  マルチSF2: {len(rendered_groups)}グループ合成完了", 87)

            # IRの準備
            hall_ir, room_ir = self._ensure_ir_files()

            # 各グループにパートEQ + 空間処理を適用してミックス
            target_len = max(len(a) for a, _ in rendered_groups.values())
            if rendered_groups and next(iter(rendered_groups.values()))[0].ndim == 2:
                mix = np.zeros((target_len, 2), dtype=np.float32)
            else:
                mix = np.zeros(target_len, dtype=np.float32)

            for gname, (audio, gcfg) in rendered_groups.items():
                # パートEQ
                audio = self._apply_part_eq(audio, gcfg.get("eq", {}), role=gname)
                # ステレオワイドニング
                width = gcfg.get("stereo_width", 1.0)
                audio = self._stereo_widen(audio, width)
                # パートリバーブ (コンボリューション)
                reverb_send = gcfg.get("reverb_send", 0.15)
                if reverb_send > 0 and hall_ir is not None:
                    ir = room_ir if gname in ("drums", "bass") else hall_ir
                    if ir is not None:
                        audio = self._convolution_reverb(audio, ir, reverb_send)
                # 長さ合わせてミックス
                n = min(len(audio), target_len)
                if audio.ndim == mix.ndim:
                    mix[:n] += audio[:n]
                elif audio.ndim == 1 and mix.ndim == 2:
                    mix[:n, 0] += audio[:n]
                    mix[:n, 1] += audio[:n]
                elif audio.ndim == 2 and mix.ndim == 1:
                    mix[:n] += audio[:n].mean(axis=1)

            return mix.astype(np.float32)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _synthesize_audio(self, midi_path, parts, drum_events, duration):
        """マルチSF2合成を試み、失敗時は単一SF2、最終フォールバックは v6 加算合成"""
        # マルチSF2
        audio = self._synthesize_multi_sf2(midi_path, duration)
        if audio is not None:
            return self._trim_to_duration(audio, duration)

        # 単一SF2フォールバック
        audio = self._synthesize_fluidsynth(midi_path, duration)
        if audio is not None:
            return self._trim_to_duration(audio, duration)

        # v6 加算合成フォールバック
        self._log("  FluidSynth 未使用、v6加算合成を使用", 84)
        n = int((duration + 0.5) * SR)
        audio = self._synth_parts(parts, n) * 0.85
        audio += self._synth_drums(drum_events, n) * 0.55
        return self._trim_to_duration(audio, duration)

    def _trim_to_duration(self, audio, duration):
        """合成音を原曲の長さ+0.2秒に切り詰める (FluidSynth のリバーブ尾を除去)"""
        target_len = int((duration + 0.2) * SR)
        if len(audio) > target_len:
            fade_len = min(int(0.05 * SR), target_len)
            audio = audio[:target_len].copy()
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            if audio.ndim == 2:
                audio[-fade_len:] *= fade[:, np.newaxis]
            else:
                audio[-fade_len:] *= fade
        return audio

    # ---- パートEQ ------------------------------------------

    ROLE_EQ = {
        'bass': {'hpf': 35, 'lpf': 200, 'boost_hz': 80, 'boost_db': 4},
        'drums': {'hpf': 50, 'boost_hz': 60, 'boost_db': 4, 'lpf': 8000},
        'melody': {'hpf': 250, 'boost_hz': 3000, 'boost_db': 2, 'boost2_hz': 8000, 'boost2_db': 1},
        'chord': {'hpf': 180, 'cut_hz': 400, 'cut_db': -3, 'boost_hz': 5000, 'boost_db': 1},
        'pad': {'hpf': 200, 'cut_hz': 1000, 'cut_db': -6, 'boost_hz': 12000, 'boost_db': 2},
        'decoration': {'hpf': 300, 'boost_hz': 5000, 'boost_db': 1.5},
        'sub_melody': {'hpf': 120, 'boost_hz': 2000, 'boost_db': 1},
    }

    def _apply_part_eq(self, audio, eq_cfg, role=None):
        """パート種別に応じた EQ を適用する (role名またはeq_cfg dictから)"""
        if role and role in self.ROLE_EQ:
            eq_cfg = self.ROLE_EQ[role]
        if not eq_cfg:
            return audio
        nyq = SR / 2

        def _process_1d(sig):
            if 'hpf' in eq_cfg:
                freq = eq_cfg['hpf']
                b, a = butter(2, min(freq / nyq, 0.99), btype='high')
                sig = filtfilt(b, a, sig).astype(np.float32)
            if 'lpf' in eq_cfg:
                freq = eq_cfg['lpf']
                b, a = butter(2, min(freq / nyq, 0.99), btype='low')
                sig = filtfilt(b, a, sig).astype(np.float32)
            if 'boost_hz' in eq_cfg and 'boost_db' in eq_cfg:
                from scipy.signal import iirpeak
                freq = eq_cfg['boost_hz']
                gain_db = eq_cfg['boost_db']
                w0 = min(freq / nyq, 0.99)
                try:
                    b, a = iirpeak(w0, Q=1.0)
                    gain = 10 ** (gain_db / 20.0)
                    filtered = filtfilt(b, a, sig).astype(np.float32)
                    sig = (sig + (filtered - sig) * (gain - 1.0)).astype(np.float32)
                except Exception:
                    pass
            if 'boost2_hz' in eq_cfg and 'boost2_db' in eq_cfg:
                from scipy.signal import iirpeak
                freq = eq_cfg['boost2_hz']
                gain_db = eq_cfg['boost2_db']
                w0 = min(freq / nyq, 0.99)
                try:
                    b, a = iirpeak(w0, Q=1.0)
                    gain = 10 ** (gain_db / 20.0)
                    filtered = filtfilt(b, a, sig).astype(np.float32)
                    sig = (sig + (filtered - sig) * (gain - 1.0)).astype(np.float32)
                except Exception:
                    pass
            if 'cut_hz' in eq_cfg and 'cut_db' in eq_cfg:
                from scipy.signal import iirpeak
                freq = eq_cfg['cut_hz']
                gain_db = eq_cfg['cut_db']
                w0 = min(freq / nyq, 0.99)
                try:
                    b, a = iirpeak(w0, Q=0.8)
                    gain = 10 ** (gain_db / 20.0)
                    filtered = filtfilt(b, a, sig).astype(np.float32)
                    sig = (sig + (filtered - sig) * (gain - 1.0)).astype(np.float32)
                except Exception:
                    pass
            # instruments.json legacy support
            for key in ('low_shelf_db', 'high_shelf_db', 'mid_db'):
                val = eq_cfg.get(key, 0)
                if abs(val) < 0.5:
                    continue
                from scipy.signal import iirpeak
                freq_map = {'low_shelf_db': 200, 'high_shelf_db': 8000,
                            'mid_db': eq_cfg.get('mid_freq_hz', 2500)}
                freq = freq_map[key]
                w0 = min(freq / nyq, 0.99)
                try:
                    b, a = iirpeak(w0, Q=0.7 if 'shelf' in key else 1.5)
                    gain = 10 ** (val / 20.0)
                    filtered = filtfilt(b, a, sig).astype(np.float32)
                    sig = (sig + (filtered - sig) * (gain - 1.0)).astype(np.float32)
                except Exception:
                    pass
            return sig

        if audio.ndim == 2:
            for ch in range(audio.shape[1]):
                audio[:, ch] = _process_1d(audio[:, ch])
        else:
            audio = _process_1d(audio)
        return audio

    # ---- ステレオワイドニング --------------------------------

    def _stereo_widen(self, audio, width=1.0):
        """M/S 方式のステレオ幅調整。width=0でモノ、1.0で変化なし、>1で広がる"""
        if audio.ndim != 2 or audio.shape[1] != 2:
            return audio
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5
        side *= width
        audio = np.column_stack([mid + side, mid - side]).astype(np.float32)
        return audio

    # ---- コンボリューション・リバーブ (FFT) -----------------

    def _ensure_ir_files(self):
        """ホール/ルーム IR ファイルを生成して返す (CC0 アルゴリズム生成)"""
        ir_dir = _ASSETS_DIR / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        hall_path = ir_dir / "hall.npy"
        room_path = ir_dir / "room.npy"

        if hall_path.exists() and room_path.exists():
            try:
                return np.load(str(hall_path)), np.load(str(room_path))
            except Exception:
                pass

        hall_ir = self._generate_synthetic_ir(
            duration=2.2, decay_time=2.0, predelay_ms=25,
            density=0.8, damping=0.4, seed=1001
        )
        room_ir = self._generate_synthetic_ir(
            duration=0.8, decay_time=0.6, predelay_ms=8,
            density=0.5, damping=0.6, seed=1002
        )
        try:
            np.save(str(hall_path), hall_ir)
            np.save(str(room_path), room_ir)
        except Exception:
            pass
        return hall_ir, room_ir

    def _generate_synthetic_ir(self, duration=2.0, decay_time=1.8,
                               predelay_ms=20, density=0.7,
                               damping=0.4, seed=42):
        """アルゴリズムによるインパルスレスポンス生成 (CC0, パブリックドメイン)。

        指数減衰ノイズ + 初期反射 + ハイカットダンピングで
        自然なリバーブテイルを構築する。ステレオ (N,2) で返す。
        """
        rng = np.random.RandomState(seed)
        n_samples = int(duration * SR)
        predelay_samples = int(predelay_ms * SR / 1000)

        ir = np.zeros((n_samples, 2), dtype=np.float32)

        # 初期反射 (6本)
        early_delays = [int(d * SR / 1000) for d in [5, 11, 17, 23, 31, 37]]
        early_gains = [0.7, 0.55, 0.45, 0.35, 0.28, 0.22]
        for i, (d, g) in enumerate(zip(early_delays, early_gains)):
            pos = predelay_samples + d
            if pos < n_samples:
                pan = 0.3 + 0.4 * (i % 2)
                ir[pos, 0] += g * (1 - pan)
                ir[pos, 1] += g * pan

        # 拡散テイル (指数減衰ノイズ)
        t = np.arange(n_samples) / SR
        envelope = np.exp(-t * (3.0 / max(decay_time, 0.1)))
        tail_start = predelay_samples + early_delays[-1] + int(0.01 * SR)
        for ch in range(2):
            noise = rng.randn(n_samples).astype(np.float32) * density * 0.3
            # ダンピング (ハイカット)
            cutoff = min((1.0 - damping) * 12000, SR / 2 - 100) / (SR / 2)
            cutoff = max(0.01, min(cutoff, 0.99))
            try:
                b, a = butter(2, cutoff, btype="low")
                noise = filtfilt(b, a, noise).astype(np.float32)
            except Exception:
                pass
            noise *= envelope
            noise[:tail_start] = 0
            ir[:, ch] += noise

        # 正規化
        peak = np.max(np.abs(ir))
        if peak > 1e-6:
            ir = ir / peak * 0.95
        return ir.astype(np.float32)

    def _convolution_reverb(self, audio, ir, wet=0.2):
        """FFT畳み込みリバーブ (CPU軽量)"""
        from scipy.signal import fftconvolve

        dry = audio.copy()
        if audio.ndim == 1:
            reverbed_l = fftconvolve(audio, ir[:, 0], mode="full")[:len(audio)]
            reverbed_r = fftconvolve(audio, ir[:, 1], mode="full")[:len(audio)]
            wet_sig = (reverbed_l + reverbed_r) * 0.5
            # サブベース除去
            try:
                b, a = butter(2, 150 / (SR / 2), btype="high")
                wet_sig = filtfilt(b, a, wet_sig).astype(np.float32)
            except Exception:
                pass
            return ((1 - wet) * audio + wet * wet_sig).astype(np.float32)
        else:
            result = dry.copy()
            for ch in range(min(audio.shape[1], 2)):
                ir_ch = ir[:, min(ch, ir.shape[1] - 1)]
                reverbed = fftconvolve(audio[:, ch], ir_ch, mode="full")[:len(audio)]
                try:
                    b, a = butter(2, 150 / (SR / 2), btype="high")
                    reverbed = filtfilt(b, a, reverbed).astype(np.float32)
                except Exception:
                    pass
                result[:, ch] = ((1 - wet) * audio[:, ch] + wet * reverbed).astype(np.float32)
            return result

    # ---- 合成 (マルチSF2 優先) -----------------------------

    def _synthesize_fluidsynth(self, midi_path, duration):
        """FluidSynth CLI で MIDI → WAV → ndarray を実行する

        pyfluidsynth (Python binding) はDLL問題が多いため、
        FluidSynth CLI バイナリを直接呼び出す方式に変更。
        Windows では自動ダウンロード対応。
        """
        # 1. SoundFont を確保
        sf2 = _find_sf2()
        if sf2 is None:
            self._log("  SoundFont を自動取得中...", 82)
            sf2 = _download_sf2(lambda m: self._log(m, 83))
        if sf2 is None:
            sf2 = _prompt_sf2_manually(lambda m: self._log(m, 83))
        if sf2 is None:
            self._log("  SoundFont が見つかりません", 84)
            return None
        sf2_name = Path(sf2).name
        try:
            sf2_size_mb = Path(sf2).stat().st_size / (1024 * 1024)
            _score = _sf2_quality_score(sf2)
            _qlabel = "高品質" if _score >= 90 else "標準" if _score >= 60 else "簡易"
            self._log(f"  SoundFont: {sf2_name} ({sf2_size_mb:.0f}MB, {_qlabel})", 84)
        except Exception:
            self._log(f"  SoundFont: {sf2_name}", 84)

        # 2. FluidSynth CLI バイナリを確保
        fs_bin = _ensure_fluidsynth_cli(lambda m: self._log(m, 84))
        if fs_bin is None:
            self._log("  FluidSynth CLI が見つかりません", 84)
            return None

        # 3. Windows 日本語パス問題回避: ASCIIのみの一時フォルダで作業
        work_dir = Path(tempfile.mkdtemp(prefix="earcopy_fs_"))
        ascii_midi = work_dir / "in.mid"
        ascii_wav = work_dir / "out.wav"
        ascii_sf2 = work_dir / "font.sf2"
        try:
            shutil.copy(midi_path, ascii_midi)
            # SF2 は容量が大きいので symlink 優先、失敗時はコピー
            if ascii_sf2.exists():
                ascii_sf2.unlink()
            try:
                ascii_sf2.symlink_to(sf2)
            except (OSError, NotImplementedError):
                shutil.copy(sf2, ascii_sf2)
        except Exception as e:
            self._log(f"  ファイルコピー失敗: {e}", -1)
            return None

        # 4. FluidSynth CLI 実行（正しいフラグ順序）
        self._log("  FluidSynth でレンダリング中 (サンプル音源使用)...", 85)
        # 重要: 全フラグを positional args (sf2, mid) の前に置く
        # タイムアウト: 曲の長さの10倍 or 最低15分
        timeout_sec = max(900, int(duration * 10))

        def _build_cmd(use_file_driver: bool) -> list:
            base = [
                fs_bin,
                "-ni",
                "-F", str(ascii_wav),
                "-r", str(SR),
                "-g", "0.85",
                "-R", "0",
                "-C", "0",
                "-T", "wav",
            ]
            if use_file_driver:
                base += ["-a", "file"]  # file-only driver (一部ビルドで非対応)
            return base + [str(ascii_sf2), str(ascii_midi)]

        def _run_fluidsynth(cmd: list):
            """(returncode, out, err) を返す。タイムアウト時は (None, None, None)。"""
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                out, err = proc.communicate(timeout=timeout_sec)
                return proc.returncode, out, err
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                self._log(f"  FluidSynth: タイムアウト ({timeout_sec}秒)", -1)
                return None, None, None

        try:
            # まず -a file ドライバで試行（WAV ファイル出力専用モード）
            rc, out, err = _run_fluidsynth(_build_cmd(use_file_driver=True))
            if rc is None:
                return None  # timed out

            # -a file 非対応ビルドでは returncode != 0 になるのでドライバ省略でリトライ
            if rc != 0:
                self._log("  -a file ドライバ非対応、再試行中...", 84)
                if ascii_wav.exists():
                    ascii_wav.unlink()
                rc, out, err = _run_fluidsynth(_build_cmd(use_file_driver=False))
                if rc is None:
                    return None

            if rc != 0:
                tail = (err or out or "no output").strip().splitlines()[-6:]
                self._log(f"  FluidSynth エラー (rc={rc}):", -1)
                for line in tail:
                    self._log(f"    {line}", -1)
                return None

            if not ascii_wav.exists() or ascii_wav.stat().st_size < 1000:
                self._log(f"  FluidSynth: WAV出力が空 ({ascii_wav.stat().st_size if ascii_wav.exists() else 0} bytes)", -1)
                if err:
                    for line in err.strip().splitlines()[-4:]:
                        self._log(f"    {line}", -1)
                return None

            audio, _ = librosa.load(str(ascii_wav), sr=SR, mono=False)
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio]).astype(np.float32)
            elif audio.ndim == 2:
                audio = audio.T.astype(np.float32)  # (2, N) → (N, 2)
            target_len = int((duration + 0.5) * SR)
            if len(audio) > target_len:
                audio = audio[:target_len]
            actual_dur = len(audio) / SR
            if actual_dur < duration * 0.5:
                self._log(f"  ⚠ FluidSynth出力が短い ({actual_dur:.1f}s / 期待{duration:.1f}s)", -1)
            self._log(f"  ✓ FluidSynth レンダリング完了 ({actual_dur:.1f}秒)", 88)
            return audio
        except Exception as e:
            self._log(f"  FluidSynth 実行エラー: {e}", -1)
            return None
        finally:
            # 一時ファイル掃除
            for p in (ascii_midi, ascii_wav):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            shutil.rmtree(work_dir, ignore_errors=True)

    # ---- マスタリング (プロダクション品質ミックスチェーン) ----
    # 原盤権侵害防止のため、原音ステムを最終出力にミックスする経路は
    # 一切設けない (旧 _hybrid_mix は v7.x で削除)。最終 WAV/MP3 は
    # FluidSynth の合成音 (synth_audio) のみで構成する。

    def _soft_compress(self, audio, threshold_db=-18.0, ratio=3.0):
        """ソフトニー・エンベロープフォロワー付きコンプレッサー"""
        threshold = 10 ** (threshold_db / 20.0)
        attack = np.exp(-1 / (SR * 0.005))
        release = np.exp(-1 / (SR * 0.05))
        env = 0.0
        n_samples = len(audio)
        gain_arr = np.ones(n_samples, dtype=np.float32)
        block = 256
        for i in range(0, n_samples - block, block):
            chunk = audio[i:i + block]
            rms_val = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
            if rms_val > env:
                env = attack * env + (1 - attack) * rms_val
            else:
                env = release * env + (1 - release) * rms_val
            if env > threshold:
                desired_gain = (threshold + (env - threshold) / ratio) / max(env, 1e-9)
            else:
                desired_gain = 1.0
            gain_arr[i:i + block] = desired_gain
        avg_gain = float(np.mean(gain_arr))
        if avg_gain > 1e-9:
            makeup_db = max(0.0, min(6.0, -20 * np.log10(avg_gain)))
        else:
            makeup_db = 3.0
        makeup = 10 ** (makeup_db / 20.0)
        gain_arr *= makeup
        if audio.ndim == 2:
            audio = audio * gain_arr[:, np.newaxis]
        else:
            audio = audio * gain_arr
        return audio.astype(np.float32)

    def _measure_lufs(self, audio):
        """ITU-R BS.1770-4 簡易 LUFS 測定 (K-weight フィルタ + ゲーティング)"""
        # K-weight stage 1: high shelf +4dB at 1681Hz
        nyq = SR / 2
        # 簡易 K-weight: 2段の IIR で近似
        try:
            from scipy.signal import iirpeak
            b1, a1 = iirpeak(1681 / nyq, Q=0.7)
            b2, a2 = butter(2, 38 / nyq, btype="high")
        except Exception:
            # フィルタ失敗時は RMS ベースの概算
            check = audio.mean(axis=1) if audio.ndim == 2 else audio
            rms = float(np.sqrt(np.mean(check ** 2) + 1e-12))
            return 20 * np.log10(max(rms, 1e-12)) - 0.691

        if audio.ndim == 2:
            channels = [audio[:, ch] for ch in range(audio.shape[1])]
        else:
            channels = [audio]

        channel_powers = []
        for ch in channels:
            # K-weight filtering
            filtered = filtfilt(b1, a1, ch).astype(np.float32)
            filtered = filtfilt(b2, a2, filtered).astype(np.float32)
            # ゲーティング: 400ms ブロック, 75%オーバーラップ
            block_samples = int(0.4 * SR)
            step = block_samples // 4
            blocks = []
            for i in range(0, len(filtered) - block_samples, step):
                block = filtered[i:i + block_samples]
                blocks.append(float(np.mean(block ** 2)))
            if not blocks:
                blocks = [float(np.mean(filtered ** 2))]
            # 絶対ゲート -70 LUFS
            abs_gate = 10 ** ((-70 + 0.691) / 10.0)
            gated = [b for b in blocks if b > abs_gate]
            if not gated:
                gated = blocks
            # 相対ゲート -10 dB below ungated
            ungated_mean = np.mean(gated)
            rel_gate = ungated_mean * 10 ** (-10 / 10.0)
            final_blocks = [b for b in gated if b > rel_gate]
            if not final_blocks:
                final_blocks = gated
            channel_powers.append(float(np.mean(final_blocks)))

        # ステレオの場合: G_l = G_r = 1.0
        mean_power = sum(channel_powers) / len(channel_powers)
        lufs = -0.691 + 10 * np.log10(max(mean_power, 1e-12))
        return float(lufs)

    def _soft_limit(self, audio, threshold_db=-1.0, lookahead_ms=5.0):
        """Lookahead付きソフトリミッター: ピークを threshold_db 以下に抑える"""
        threshold = 10 ** (threshold_db / 20.0)
        lookahead = int(lookahead_ms * SR / 1000)

        if audio.ndim == 2:
            peak_env = np.max(np.abs(audio), axis=1)
        else:
            peak_env = np.abs(audio)

        gain = np.ones(len(peak_env), dtype=np.float32)
        for i in range(len(peak_env)):
            if peak_env[i] > threshold:
                gain[i] = threshold / peak_env[i]

        if lookahead > 0:
            smoothed = np.copy(gain)
            for i in range(len(gain) - 1, -1, -1):
                end = min(i + lookahead, len(gain))
                smoothed[i] = np.min(gain[i:end])
            attack_coeff = np.exp(-1 / max(lookahead, 1))
            env = 1.0
            for i in range(len(smoothed)):
                if smoothed[i] < env:
                    env = smoothed[i]
                else:
                    env = attack_coeff * env + (1 - attack_coeff) * smoothed[i]
                smoothed[i] = env
            gain = smoothed

        if audio.ndim == 2:
            audio = audio * gain[:, np.newaxis]
        else:
            audio = audio * gain
        return audio.astype(np.float32)

    def _true_peak_limit(self, audio, ceiling_dbtp=-1.0):
        """True Peak リミッター: 4倍オーバーサンプリングでISPを検出・抑制"""
        ceiling = 10 ** (ceiling_dbtp / 20.0)
        from scipy.signal import resample_poly

        def _limit_1d(sig):
            up = resample_poly(sig, 4, 1).astype(np.float32)
            peak_up = np.max(np.abs(up))
            if peak_up <= ceiling:
                return sig
            block = 4 * 16
            attack_samples = max(1, int(0.0001 * SR * 4))
            release_coeff = np.exp(-1.0 / (0.05 * SR * 4))
            gain_curve = np.ones(len(up), dtype=np.float32)
            for i in range(0, len(up) - block, block):
                chunk_peak = np.max(np.abs(up[i:i + block]))
                if chunk_peak > ceiling:
                    g = ceiling / chunk_peak
                    start = max(0, i - attack_samples)
                    gain_curve[start:i + block] = np.minimum(
                        gain_curve[start:i + block], g
                    )
            env = 1.0
            for i in range(len(gain_curve)):
                if gain_curve[i] < env:
                    env = gain_curve[i]
                else:
                    env = release_coeff * env + (1 - release_coeff) * gain_curve[i]
                gain_curve[i] = env
            up *= gain_curve
            return resample_poly(up, 1, 4).astype(np.float32)[:len(sig)]

        if audio.ndim == 2:
            for ch in range(audio.shape[1]):
                audio[:, ch] = _limit_1d(audio[:, ch])
        else:
            audio = _limit_1d(audio)
        return audio.astype(np.float32)

    def _lufs_normalize(self, audio, target_lufs=-14.0):
        """LUFS ベースのラウドネス正規化 (YouTube 基準 -14 LUFS)"""
        current_lufs = self._measure_lufs(audio)
        diff_db = target_lufs - current_lufs
        # 極端な補正は避ける
        diff_db = max(-20.0, min(20.0, diff_db))
        gain = 10 ** (diff_db / 20.0)
        audio = audio * gain
        audio = np.clip(audio, -0.999, 0.999).astype(np.float32)
        return audio

    def _master(self, audio):
        """プロダクション品質マスタリングチェーン:
        DC除去 → HP/LP → コンプ (-18dB, 3:1) → ソフトリミット → -14 LUFS 正規化。
        モノ (N,) とステレオ (N,2) の両方に対応。
        """
        if len(audio) == 0:
            return audio.astype(np.float32)

        is_stereo = audio.ndim == 2
        audio = (audio - np.mean(audio, axis=0)).astype(np.float32)
        nyq = SR / 2

        # ハイパス 35Hz
        b, a = butter(3, 35 / nyq, btype="high")
        if is_stereo:
            for ch in range(audio.shape[1]):
                audio[:, ch] = filtfilt(b, a, audio[:, ch]).astype(np.float32)
        else:
            audio = filtfilt(b, a, audio).astype(np.float32)

        # ローパス 18kHz
        b, a = butter(2, min(18000 / nyq, 0.99), btype="low")
        if is_stereo:
            for ch in range(audio.shape[1]):
                audio[:, ch] = filtfilt(b, a, audio[:, ch]).astype(np.float32)
        else:
            audio = filtfilt(b, a, audio).astype(np.float32)

        # マスター低域ブースト: +1.5dB @ 100Hz (ベース帯不足解消)
        try:
            from scipy.signal import iirpeak
            w0_lo = min(100 / nyq, 0.99)
            b_lo, a_lo = iirpeak(w0_lo, Q=0.7)
            gain_lo = 10 ** (1.5 / 20.0)
            if is_stereo:
                for ch in range(audio.shape[1]):
                    filt = filtfilt(b_lo, a_lo, audio[:, ch]).astype(np.float32)
                    audio[:, ch] = (audio[:, ch] + (filt - audio[:, ch]) * (gain_lo - 1.0)).astype(np.float32)
            else:
                filt = filtfilt(b_lo, a_lo, audio).astype(np.float32)
                audio = (audio + (filt - audio) * (gain_lo - 1.0)).astype(np.float32)
        except Exception:
            pass

        # マスターエア感: +1dB @ 12kHz
        try:
            w0_air = min(12000 / nyq, 0.99)
            b_air, a_air = iirpeak(w0_air, Q=0.7)
            gain_air = 10 ** (1.0 / 20.0)
            if is_stereo:
                for ch in range(audio.shape[1]):
                    filt = filtfilt(b_air, a_air, audio[:, ch]).astype(np.float32)
                    audio[:, ch] = (audio[:, ch] + (filt - audio[:, ch]) * (gain_air - 1.0)).astype(np.float32)
            else:
                filt = filtfilt(b_air, a_air, audio).astype(np.float32)
                audio = (audio + (filt - audio) * (gain_air - 1.0)).astype(np.float32)
        except Exception:
            pass

        # パートごとゲイン: 合計時にピーク<1.0 を目標
        peak = np.max(np.abs(audio))
        if peak > 0.8:
            audio = (audio * (0.8 / peak)).astype(np.float32)

        # ソフトコンプレッション (閾値 -18dB, ratio 3:1)
        audio = self._soft_compress(audio, threshold_db=-18.0, ratio=3.0)

        # ソフトリミッター (-1dB threshold, lookahead 5ms)
        audio = self._soft_limit(audio, threshold_db=-1.0, lookahead_ms=5.0)

        # -14 LUFS ラウドネス正規化 (YouTube 基準)
        audio = self._lufs_normalize(audio, target_lufs=-14.0)

        # True Peak リミッター (-1 dBTP、MP3エンコード後のISP防止)
        audio = self._true_peak_limit(audio, ceiling_dbtp=-1.0)

        # 安全網: ハードクリップ
        audio = np.clip(audio, -0.891, 0.891).astype(np.float32)

        return audio

    def _validate_output(self, audio, duration):
        """出力音声の品質チェック。モノ/ステレオ両対応。"""
        issues = []
        # ステレオの場合はチャンネル平均でチェック
        check = audio.mean(axis=1) if audio.ndim == 2 else audio
        fmt = "ステレオ" if audio.ndim == 2 else "モノラル"

        actual_dur = len(check) / SR
        if actual_dur < duration * 0.9:
            issues.append(f"出力が短い ({actual_dur:.1f}s / 期待 {duration:.1f}s)")
        elif actual_dur > duration * 1.2:
            issues.append(f"出力が長い ({actual_dur:.1f}s / 期待 {duration:.1f}s)")
        rms = float(np.sqrt(np.mean(check ** 2)))
        if rms < 0.005:
            issues.append(f"音量が極端に小さい (RMS={rms:.4f})")
        clip_count = int(np.sum(np.abs(audio) > 0.99))
        clip_ratio = clip_count / max(audio.size, 1)
        if clip_ratio > 0.001:
            issues.append(f"クリッピング検出 ({clip_count}サンプル, {clip_ratio*100:.2f}%)")
        try:
            check_len = min(len(check), SR * 5)
            fft = np.abs(np.fft.rfft(check[:check_len]))
            freqs = np.fft.rfftfreq(check_len, 1 / SR)
            lo = fft[freqs < 200].sum()
            mid = fft[(freqs >= 200) & (freqs < 4000)].sum()
            hi = fft[freqs >= 4000].sum()
            total = lo + mid + hi + 1e-9
            if mid / total < 0.15:
                issues.append("中域(200-4kHz)のエネルギーが不足")
            if lo / total < 0.05:
                issues.append("低域(〜200Hz)のエネルギーが不足")
        except Exception:
            pass
        dc = float(np.mean(audio))
        if abs(dc) > 0.01:
            issues.append(f"DCオフセット検出 ({dc:.4f})")
        if issues:
            self._log(f"⚠ 出力品質チェック ({fmt}) — 以下の問題を検出:", 92)
            for iss in issues:
                self._log(f"  • {iss}", 92)
        else:
            self._log(f"✓ 出力品質チェック OK ({fmt}, 長さ・音量・スペクトル正常)", 92)
        return len(issues) == 0

    def _freq_balance_report(self, output_audio, input_audio, output_path):
        """出力と原曲の周波数バランスを比較し、JSONレポートを生成"""
        import json as _json
        bands = [
            ("sub", 0, 80),
            ("bass", 80, 250),
            ("low_mid", 250, 1000),
            ("mid", 1000, 4000),
            ("high_mid", 4000, 8000),
            ("high", 8000, 16000),
        ]
        def _band_energy(audio_1d):
            n = min(len(audio_1d), SR * 10)
            fft = np.abs(np.fft.rfft(audio_1d[:n]))
            freqs = np.fft.rfftfreq(n, 1 / SR)
            total = float(np.sum(fft ** 2)) + 1e-12
            result = {}
            for name, lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                energy = float(np.sum(fft[mask] ** 2))
                result[name] = round(energy / total * 100, 2)
            return result

        out_mono = output_audio.mean(axis=1) if output_audio.ndim == 2 else output_audio
        in_mono = input_audio if input_audio.ndim == 1 else input_audio.mean(axis=1)
        report = {
            "output": _band_energy(out_mono),
            "input": _band_energy(in_mono),
            "ratio": {},
        }
        for name, _, _ in bands:
            inp = max(report["input"][name], 0.01)
            report["ratio"][name] = round(report["output"][name] / inp, 2)
        report_path = str(Path(output_path).with_suffix(".freq_report.json"))
        try:
            Path(report_path).write_text(
                _json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._log(f"  周波数バランスレポート: {report_path}", 93)
        except Exception:
            pass
        return report

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
        # テンポグラムで候補BPMを推定し、音楽的に一般的な範囲(60-180)に補正
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP,
                                                  aggregate=np.median)
        tempo_candidates = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                               hop_length=HOP)
        start_bpm = float(np.atleast_1d(tempo_candidates)[0])
        if start_bpm > 200:
            start_bpm /= 2.0
        elif start_bpm < 50:
            start_bpm *= 2.0

        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP,
            start_bpm=start_bpm, tightness=100
        )
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
        tempo_val = float(np.atleast_1d(tempo)[0])
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
            mid_lo = fft[(freqs >= 120) & (freqs < 200)].sum() / tot
            mid = fft[(freqs >= 200) & (freqs < 1500)].sum() / tot
            hi = fft[freqs >= 3000].sum() / tot
            if lo > 0.45:
                kind = 'kick'
            elif hi > 0.55:
                kind = 'hihat'
            elif hi > 0.35 and mid < 0.2:
                kind = 'open_hat'
            elif mid > 0.3 and hi > 0.15:
                kind = 'ride'
            elif mid_lo > 0.25 and lo < 0.35:
                kind = 'tom'
            else:
                kind = 'snare'
            peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
            vel = int(np.clip(60 + peak * 120, 40, 127))
            events.append((ot, kind, vel))
        return events

    # ---- v7.0 物理モデリングシンセサイザー ====================

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

    # ---- 合成ユーティリティ --------------------------------

    def _karplus_strong(self, freq, duration, decay=0.996, brightness=0.5):
        """Karplus-Strong 物理弦モデル（lfilter による高速実装）"""
        from scipy.signal import lfilter as _lf
        N = max(2, int(SR / freq))
        n_samples = max(1, int(duration * SR))
        excitation = np.zeros(n_samples)
        noise = np.random.uniform(-1, 1, N)
        filt_w = max(0.1, 1.0 - brightness)
        noise = np.convolve(noise, [filt_w / 2, 1 - filt_w, filt_w / 2], mode='same')
        excitation[:N] = noise
        a = np.zeros(N + 2)
        a[0] = 1.0
        a[N] = -decay * 0.5
        a[N + 1] = -decay * 0.5
        return _lf(np.array([1.0]), a, excitation)

    def _fm_synth(self, freq, duration, mod_ratio=2.0, mod_index=3.0, decay=2.0):
        """FM合成: mod_index が音色の豊かさを決定"""
        t = self._t(duration)
        mod = mod_index * np.sin(2 * np.pi * freq * mod_ratio * t)
        mod *= np.exp(-t * decay)
        return np.sin(2 * np.pi * freq * t + mod)

    def _bowed_string(self, freq, duration, vibrato_rate=5.5, vibrato_depth=0.004):
        """弓弦シミュレーション: 鉤型波+フォルマント+遅延ビブラート"""
        t = self._t(duration)
        vib_onset = np.clip(t - 0.15, 0, None) * 3
        vib_onset = np.minimum(vib_onset, 1.0)
        vib = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t) * vib_onset
        phase = 2 * np.pi * freq * vib * t
        saw = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        # 2次 LP で粗いエッジを丸める (lfilter)
        from scipy.signal import lfilter as _lf
        cutoff = min(freq * 6, SR * 0.45)
        rc = 1 / (2 * np.pi * cutoff)
        dt = 1 / SR
        alpha = dt / (rc + dt)
        b = np.array([alpha * alpha])
        a = np.array([1.0, -2 * (1 - alpha), (1 - alpha) ** 2])
        return _lf(b, a, saw)

    def _reverb_signal(self, signal, room_size=0.65, wet=0.18):
        """Freeverb風 Schroeder リバーブ (4comb + 2allpass, lfilter 高速実装)"""
        from scipy.signal import lfilter as _lf
        comb_params = [
            (1557, room_size * 0.93),
            (1617, room_size * 0.91),
            (1491, room_size * 0.88),
            (1422, room_size * 0.86),
        ]
        tail = int(SR * 0.6)
        padded = np.zeros(len(signal) + tail)
        padded[:len(signal)] = signal

        reverbed = np.zeros(len(padded))
        for delay, gain in comb_params:
            a = np.zeros(delay + 1)
            a[0] = 1.0
            a[delay] = -gain
            reverbed += _lf(np.array([1.0]), a, padded)
        reverbed /= len(comb_params)

        allpass_params = [(int(0.089 * SR), 0.5), (int(0.006 * SR), 0.5)]
        for delay, gain in allpass_params:
            b = np.zeros(delay + 1)
            b[0] = -gain
            b[delay] = 1.0
            a = np.zeros(delay + 1)
            a[0] = 1.0
            a[delay] = -gain
            reverbed = _lf(b, a, reverbed)

        n = len(signal)
        # サブベースのリバーブ除去（150Hz以下は残響させない）
        try:
            nyq = SR / 2
            b_hp, a_hp = butter(2, 150 / nyq, btype='high')
            wet_sig = filtfilt(b_hp, a_hp, reverbed[:n]).astype(np.float32)
        except Exception:
            wet_sig = reverbed[:n].astype(np.float32)
        return ((1 - wet) * signal + wet * wet_sig).astype(np.float32)

    # ---- 15楽器 物理モデリング音色 ==========================

    def _tone_piano(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        # Karplus-Strong に明るいアタック + デチューンした複弦
        decay = 0.997 - hz / 80000
        s1 = self._karplus_strong(hz, dur, decay, brightness=0.7)
        s2 = self._karplus_strong(hz * 1.001, dur, decay, brightness=0.6)
        s3 = self._karplus_strong(hz * 0.999, dur, decay, brightness=0.6)
        sig = s1 * 0.5 + s2 * 0.25 + s3 * 0.25
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        return sig * self._env(t, 0.002, 0.06, 0.55, 0.1) * (vel / 127) * 0.30

    def _tone_e_piano(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        # FM合成: DX7風のベルっぽいエレピ
        sig = self._fm_synth(hz, dur, mod_ratio=14.0, mod_index=vel / 127 * 4.0, decay=3.5)
        sig += self._fm_synth(hz, dur, mod_ratio=1.0, mod_index=2.0, decay=2.0) * 0.4
        t = self._t(dur)
        return sig * np.exp(-t * 2.2) * self._env(t, 0.001, 0.08, 0.4, 0.12) * (vel / 127) * 0.22

    def _tone_glockenspiel(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(max(dur, 1.0))
        # 非整数倍音で金属質
        sig = (np.sin(2 * np.pi * hz * t)
               + 0.5 * np.sin(2 * np.pi * hz * 2.76 * t)
               + 0.3 * np.sin(2 * np.pi * hz * 5.4 * t)
               + 0.15 * np.sin(2 * np.pi * hz * 8.93 * t))
        return sig * np.exp(-t * 2.5) * self._env(t, 0.0005, 0.01, 0.25, 0.3) * (vel / 127) * 0.14

    def _tone_organ(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        # ハモンドオルガン風: ドローバー合成
        drawbars = [0.8, 1.0, 0.6, 0.8, 0.3, 0.5, 0.2, 0.1, 0.05]
        ratios = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
        sig = sum(d * np.sin(2 * np.pi * hz * r * t) for d, r in zip(drawbars, ratios) if hz * r < SR / 2)
        # ロータリースピーカー風トレモロ
        trem = 1 + 0.15 * np.sin(2 * np.pi * 6.5 * t)
        return sig / 4 * trem * self._env(t, 0.008, 0.02, 0.88, 0.04) * (vel / 127) * 0.16

    def _tone_guitar_nylon(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        sig = self._karplus_strong(hz, dur, decay=0.994, brightness=0.3)
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        # ボディ共鳴: 低域ブースト
        from scipy.signal import lfilter as _lf
        b_r, a_r = butter(2, min(2500 / (SR / 2), 0.99), btype='low')
        sig = _lf(b_r, a_r, sig)
        return sig * self._env(t, 0.002, 0.04, 0.35, 0.06) * (vel / 127) * 0.22

    def _tone_guitar_clean(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        sig = self._karplus_strong(hz, dur, decay=0.995, brightness=0.6)
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        return sig * self._env(t, 0.001, 0.03, 0.4, 0.08) * (vel / 127) * 0.20

    def _tone_bass(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        # KS + サブ基音
        sig = self._karplus_strong(hz, dur, decay=0.993, brightness=0.25)
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        sub = np.sin(2 * np.pi * hz * t) * np.exp(-t * 1.2)
        sig = sig * 0.6 + sub * 0.4
        return sig * self._env(t, 0.005, 0.04, 0.7, 0.08) * (vel / 127) * 0.38

    def _tone_violin(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        sig = self._bowed_string(hz, dur, vibrato_rate=5.8, vibrato_depth=0.005)
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        return sig * self._env(t, 0.04, 0.06, 0.85, 0.1) * (vel / 127) * 0.22

    def _tone_viola(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        sig = self._bowed_string(hz, dur, vibrato_rate=5.2, vibrato_depth=0.004)
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        return sig * self._env(t, 0.05, 0.08, 0.8, 0.12) * (vel / 127) * 0.20

    def _tone_cello(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        sig = self._bowed_string(hz, dur, vibrato_rate=4.5, vibrato_depth=0.004)
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        return sig * self._env(t, 0.03, 0.08, 0.82, 0.12) * (vel / 127) * 0.24

    def _tone_strings(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        # 3本のデチューン弓弦でアンサンブル
        s1 = self._bowed_string(hz, dur, 5.0, 0.003)
        s2 = self._bowed_string(hz * 1.002, dur, 5.3, 0.003)
        s3 = self._bowed_string(hz * 0.998, dur, 4.7, 0.003)
        sig = (s1 + s2 + s3) / 3
        t = self._t(dur)
        if len(sig) > len(t):
            sig = sig[:len(t)]
        elif len(sig) < len(t):
            sig = np.pad(sig, (0, len(t) - len(sig)))
        return sig * self._env(t, 0.1, 0.12, 0.7, 0.18) * (vel / 127) * 0.16

    def _tone_choir(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        vib = 1 + 0.005 * np.sin(2 * np.pi * 5.5 * t)
        # フォルマント3バンドでアー母音
        formants = [(800, 80), (1200, 90), (2500, 120)]
        sig = np.zeros(len(t))
        for ff, bw in formants:
            carrier = np.sin(2 * np.pi * hz * vib * t)
            env_f = np.exp(-0.5 * ((hz - ff) / bw) ** 2) + 0.1
            sig += carrier * env_f
        # デチューン
        sig += 0.3 * np.sin(2 * np.pi * hz * 1.003 * vib * t)
        sig += 0.3 * np.sin(2 * np.pi * hz * 0.997 * vib * t)
        return sig / 3 * self._env(t, 0.18, 0.12, 0.6, 0.22) * (vel / 127) * 0.13

    def _tone_trumpet(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        vib = 1 + 0.002 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t - 0.1, 0, 1)
        # 矩形波に近い奇数倍音構成
        sig = np.zeros(len(t))
        for h in range(1, 12, 2):
            if hz * h >= SR / 2:
                break
            sig += (1.0 / h) * np.sin(2 * np.pi * hz * h * vib * t)
        sig += 0.2 * np.random.randn(len(t)) * np.exp(-t * 15)  # ブレスノイズ
        return sig * self._env(t, 0.025, 0.05, 0.85, 0.06) * (vel / 127) * 0.20

    def _tone_flute(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        vib = 1 + 0.003 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t - 0.2, 0, 1)
        sig = np.sin(2 * np.pi * hz * vib * t) + 0.08 * np.sin(2 * np.pi * hz * 2 * vib * t)
        # ブレスノイズ
        breath = np.random.randn(len(t)) * 0.05
        breath *= self._env(t, 0.08, 0.1, 0.03, 0.05)
        sig += breath
        return sig * self._env(t, 0.05, 0.08, 0.8, 0.12) * (vel / 127) * 0.18

    def _tone_pad(self, midi, dur, vel):
        hz = 440 * SEMI ** (midi - 69)
        t = self._t(dur)
        # ゆっくり揺れるデチューン和音
        s1 = np.sin(2 * np.pi * hz * t)
        s2 = np.sin(2 * np.pi * hz * 1.004 * t)
        s3 = np.sin(2 * np.pi * hz * 0.996 * t)
        lfo = 1 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
        return (s1 + s2 + s3) / 3 * lfo * self._env(t, 0.2, 0.25, 0.6, 0.35) * (vel / 127) * 0.13

    # Old 15-instrument tone map (before consolidation):
    # TONE_FN = {
    #     'piano':'_tone_piano', 'e_piano':'_tone_e_piano',
    #     'glockenspiel':'_tone_glockenspiel', 'organ':'_tone_organ',
    #     'guitar_nylon':'_tone_guitar_nylon', 'guitar_clean':'_tone_guitar_clean',
    #     'bass':'_tone_bass', 'violin':'_tone_violin', 'viola':'_tone_viola',
    #     'cello':'_tone_cello', 'strings':'_tone_strings', 'choir':'_tone_choir',
    #     'trumpet':'_tone_trumpet', 'flute':'_tone_flute', 'pad':'_tone_pad',
    # }
    TONE_FN = {
        'melody': '_tone_piano',
        'chord': '_tone_strings',
        'bass': '_tone_bass',
        'pad': '_tone_strings',
        'decoration': '_tone_glockenspiel',
        'sub_melody': '_tone_cello',
    }

    def _synth_parts(self, parts, n):
        buf = np.zeros(n)
        for name, evts in parts.items():
            fn = getattr(self, self.TONE_FN.get(name, '_tone_piano'))
            g = self.GAIN.get(name, 0.5)
            for (t, dur, midi, vel) in evts:
                s0 = int(t * SR)
                try:
                    tone = fn(midi, dur, vel)
                except Exception as e:
                    self._log(f"  ⚠ 音色生成スキップ [{name}] midi={midi}: {e}", -1)
                    continue
                s1 = s0 + len(tone)
                if s1 <= n:
                    buf[s0:s1] += tone * g
        # 全楽器合成後にリバーブ適用
        buf = self._reverb_signal(buf, room_size=0.55, wet=0.14)
        return buf

    # ---- ドラム合成（改良版・6種） ---------------------------

    def _synth_drums(self, events, n):
        buf = np.zeros(n)
        for evt in events:
            onset = evt[0]
            kind = evt[1]
            vel = evt[2] if len(evt) > 2 else None
            s0 = int(onset * SR)
            snd = self._drum_sound(kind)
            if vel is not None:
                snd = snd * (vel / 100.0)
            s1 = s0 + len(snd)
            if s1 <= n:
                buf[s0:s1] += snd
        # ドラムには控えめなルームリバーブ
        buf = self._reverb_signal(buf, room_size=0.35, wet=0.08)
        return buf

    def _drum_sound(self, kind):
        # テンポに応じてドラムの持続時間を調整（速い曲は短く、遅い曲は長く）
        tempo = getattr(self, '_current_tempo', 120.0)
        scale = max(0.5, min(1.5, 120.0 / max(tempo, 60.0)))
        if kind == 'kick':
            t = self._t(0.35 * scale)
            sweep = 80 * np.exp(-t * 20) + 45
            sig = np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-t * 8)
            sub = np.sin(2 * np.pi * 50 * t) * np.exp(-t * 12)
            click = np.random.randn(len(t)) * np.exp(-t * 60) * 0.15
            return (sig * 0.7 + sub * 0.25 + click) * 0.75
        elif kind == 'snare':
            t = self._t(0.2 * scale)
            body = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 20)
            body += np.sin(2 * np.pi * 330 * t) * np.exp(-t * 25) * 0.4
            noise = np.random.randn(len(t)) * np.exp(-t * 16)
            b, a = butter(2, [min(1500 / (SR / 2), 0.99), min(8000 / (SR / 2), 0.99)], btype='band')
            noise = filtfilt(b, a, noise)
            return (body * 0.45 + noise * 0.55) * 0.6
        elif kind == 'ride':
            t = self._t(0.5 * scale)
            n_ = np.random.randn(len(t))
            b, a = butter(3, min(3500 / (SR / 2), 0.99), btype='high')
            sig = filtfilt(b, a, n_) * np.exp(-t * 5)
            bell = np.sin(2 * np.pi * 2800 * t) * np.exp(-t * 8) * 0.15
            return (sig + bell) * 0.22
        elif kind == 'hihat':
            t = self._t(0.06 * scale)
            n_ = np.random.randn(len(t))
            b, a = butter(3, min(6000 / (SR / 2), 0.99), btype='high')
            return filtfilt(b, a, n_) * np.exp(-t * 70) * 0.35
        elif kind == 'open_hat':
            t = self._t(0.18 * scale)
            n_ = np.random.randn(len(t))
            b, a = butter(3, min(5000 / (SR / 2), 0.99), btype='high')
            return filtfilt(b, a, n_) * np.exp(-t * 20) * 0.32
        elif kind == 'tom':
            t = self._t(0.25 * scale)
            sweep = 120 * np.exp(-t * 15) + 60
            sig = np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-t * 12)
            return sig * 0.65
        else:
            t = self._t(0.12 * scale)
            n_ = np.random.randn(len(t))
            b, a = butter(3, min(5000 / (SR / 2), 0.99), btype='high')
            return filtfilt(b, a, n_) * np.exp(-t * 30) * 0.3

    # ---- ファイル保存 ------------------------------------

    def _save_mp3(self, audio, path):
        if not _FFMPEG_OK:
            raise RuntimeError(
                "ffmpeg が見つかりませんでした。\n"
                "ツールを再起動して ffmpeg.exe の場所を選択してください。"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            audio = np.clip(audio, -0.999, 0.999).astype(np.float32)
            sf.write(tmp, audio, SR)
            seg = AudioSegment.from_wav(tmp)
            seg.export(path, format="mp3", bitrate="320k")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ---- パート別 MIDI パン・ボリューム・エフェクト設定 -----

    PART_MIX = {
        #               pan(0=L,64=C,127=R)  vol  reverb  sustain
        'melody':        (64,  105,  55,  True),
        'chord':         (75,   85,  65,  False),
        'bass':          (64,  110,  20,  False),
        'pad':           (64,   55,  80,  False),
        'decoration':    (85,   60,  70,  False),
        'sub_melody':    (50,   80,  60,  False),
    }

    def _save_midi(self, parts, drums, tempo, path):
        """プロ品質 MIDI 出力: pan/volume/reverb/sustain を各パートに設定"""
        mid = MidiFile(type=1, ticks_per_beat=480)
        tpb = 480
        us = int(60_000_000 / tempo)
        def s2t(s): return int(s * tempo / 60 * tpb)

        used_channels = set()
        for name in parts:
            if name in self.MIDI_MAP:
                ch, _ = self.MIDI_MAP[name]
                if ch == 9:
                    self._log(f"  ⚠ MIDI_MAP '{name}' がドラム専用ch9を使用", -1)
                used_channels.add(ch)
        active_parts = sum(1 for v in parts.values() if v)
        self._log(f"  MIDI出力: {active_parts}パート, ch={sorted(used_channels)}", 76)

        # テンポトラック
        tt = MidiTrack()
        tt.append(MetaMessage('set_tempo', tempo=us, time=0))
        mid.tracks.append(tt)

        for name, evts in parts.items():
            if not evts:
                continue
            ch, prog = self.MIDI_MAP[name]
            pan, vol, reverb_amt, use_sustain = self.PART_MIX.get(
                name, (64, 85, 50, False))

            evs = []
            # Program Change + 初期CC設定
            evs.append((0, Message('program_change', channel=ch, program=prog, time=0)))
            evs.append((0, Message('control_change', channel=ch, control=7, value=vol, time=0)))   # CC7 Volume
            evs.append((0, Message('control_change', channel=ch, control=10, value=pan, time=0)))  # CC10 Pan
            evs.append((0, Message('control_change', channel=ch, control=91, value=reverb_amt, time=0)))  # CC91 Reverb
            evs.append((0, Message('control_change', channel=ch, control=93, value=35, time=0)))   # CC93 Chorus

            # サスティンペダル（ピアノ系のみ: 各ビートの頭で踏み替え）
            if use_sustain and len(evts) > 0:
                max_t = max(t + dur for t, dur, _, _ in evts)
                pedal_interval = 60.0 / max(tempo, 60) * 2  # 2ビートごとに踏み替え
                t_cur = 0.0
                while t_cur < max_t:
                    tick = s2t(t_cur)
                    evs.append((tick, Message('control_change', channel=ch, control=64, value=0, time=0)))
                    evs.append((tick + 5, Message('control_change', channel=ch, control=64, value=127, time=0)))
                    t_cur += pedal_interval
                final_tick = s2t(max_t + 0.1)
                evs.append((final_tick, Message('control_change', channel=ch, control=64, value=0, time=0)))

            # ノートイベント
            for (t, dur, midi_note, vel) in evts:
                note = int(np.clip(midi_note, 0, 127))
                v = min(max(vel, 1), 127)
                t0, t1 = s2t(t), s2t(t + dur)
                evs.append((t0, Message('note_on', channel=ch, note=note, velocity=v, time=0)))
                evs.append((t1, Message('note_off', channel=ch, note=note, velocity=0, time=0)))

            # ソート＆delta time 変換
            trk = MidiTrack()
            evs.sort(key=lambda x: x[0])
            prev = 0
            for tick, msg in evs:
                msg.time = max(0, tick - prev)
                trk.append(msg)
                prev = tick
            mid.tracks.append(trk)

        # ドラムトラック (ch9)
        DM = {'kick': 36, 'snare': 38, 'hihat': 42, 'open_hat': 46,
              'ride': 51, 'tom': 47}
        devs = []
        # ドラム初期設定
        devs.append((0, Message('control_change', channel=9, control=7, value=105, time=0)))  # Vol
        devs.append((0, Message('control_change', channel=9, control=10, value=64, time=0)))  # Pan center
        devs.append((0, Message('control_change', channel=9, control=91, value=30, time=0)))  # Reverb
        for evt in drums:
            t = evt[0]
            kind = evt[1]
            if len(evt) > 2:
                dv = int(np.clip(evt[2], 1, 127))
            else:
                dv = {'kick': 105, 'snare': 100, 'hihat': 75, 'open_hat': 80,
                      'ride': 70, 'tom': 95}.get(kind, 90)
            n = DM.get(kind, 38)
            t0 = s2t(t)
            devs.append((t0, Message('note_on', channel=9, note=n, velocity=dv, time=0)))
            devs.append((t0 + 30, Message('note_off', channel=9, note=n, velocity=0, time=0)))
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

    _AGREE_FILE = Path.home() / ".mimikopi_agreed"

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
        self._busy     = False

        if not self._check_agreement():
            self.root.destroy()
            return

        self._build()

    # ---- 免責同意ダイアログ --------------------------------

    def _check_agreement(self):
        if self._AGREE_FILE.exists():
            return True

        DISCLAIMER = (
            "【利用上の注意・免責事項】\n\n"
            "本ツールは、入力された音楽ファイルをAIで解析し、\n"
            "MIDIデータに変換した上でサンプル音源により再合成します。\n"
            "出力に原曲の録音物（原盤）は一切含まれません。\n\n"
            "ただし、生成された楽曲を公開・配信する場合、\n"
            "楽曲の著作権（作詞・作曲）に関する許諾が\n"
            "別途必要になる場合があります。\n\n"
            "● JASRAC / NexTone 管理楽曲の場合:\n"
            "  YouTube・ニコニコ動画等の包括契約対象サービスでは\n"
            "  追加手続き不要で「歌ってみた」等に利用できます。\n"
            "  それ以外のサービスでは個別の許諾申請が必要です。\n\n"
            "● 上記以外の楽曲（海外楽曲・自主制作等）:\n"
            "  権利者への直接確認が必要です。\n\n"
            "本ツールの利用により生じた著作権上の問題について、\n"
            "開発者は一切の責任を負いません。\n"
            "利用者ご自身の責任でご使用ください。"
        )

        dlg = tk.Toplevel(self.root)
        dlg.title("利用規約への同意")
        dlg.geometry("520x480")
        dlg.configure(bg=self.BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

        agreed = tk.BooleanVar(value=False)

        tk.Label(dlg, text="ご利用前にお読みください",
                 font=("Helvetica", 14, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(pady=(16, 8))

        txt = tk.Text(dlg, wrap="word", bg=self.BG2, fg=self.FG,
                      font=("Helvetica", 10), relief="flat",
                      padx=12, pady=10, height=18)
        txt.insert("1.0", DISCLAIMER)
        txt.configure(state="disabled")
        txt.pack(padx=20, fill="both", expand=True)

        def _on_agree():
            agreed.set(True)
            try:
                self._AGREE_FILE.write_text(
                    f"agreed={__import__('datetime').datetime.now().isoformat()}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
            dlg.destroy()

        def _on_decline():
            agreed.set(False)
            dlg.destroy()

        bf = tk.Frame(dlg, bg=self.BG)
        bf.pack(pady=(10, 16))
        tk.Button(bf, text="同意して利用する", command=_on_agree,
                  bg="#0f3460", fg=self.FG, font=("Helvetica", 11, "bold"),
                  relief="flat", padx=20, pady=6, cursor="hand2"
                  ).pack(side="left", padx=8)
        tk.Button(bf, text="同意しない", command=_on_decline,
                  bg=self.BG2, fg=self.FG2, font=("Helvetica", 10),
                  relief="flat", padx=14, pady=6, cursor="hand2"
                  ).pack(side="left", padx=8)

        dlg.wait_window()
        return agreed.get()

    # ---- UI構築 -----------------------------------------

    def _build(self):
        r = self.root

        # タイトル
        tk.Label(r, text="耳コピ自動生成ツール v7.0",
                 font=("Helvetica", 20, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(pady=(20, 4))
        tk.Label(r, text="Demucs × Basic Pitch × FluidSynth で原曲忠実な耳コピ",
                 font=("Helvetica", 9), bg=self.BG, fg=self.FG2).pack()
        tk.Label(r, text="※ 100% MIDI再合成・原盤不使用 (原曲音源は出力に含まれません)",
                 font=("Helvetica", 8), bg=self.BG, fg=self.FG2).pack()

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
            ("ai_inst", "AI耳コピ・ガイドメロディなし (カラオケ伴奏向け)"),
        ]
        for val, label in modes:
            tk.Radiobutton(
                mf, text=label, variable=self._mode, value=val,
                bg=self.BG, fg=self.FG, selectcolor=self.BG2,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=("Helvetica", 9), anchor="w",
            ).pack(anchor="w", padx=8, pady=1)

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

        mode = self._mode.get()

        def worker():
            engine = EarCopyEngine(on_progress=self._on_prog, mode=mode)
            ok, res = engine.process(path, output)
            self.root.after(0, lambda: (self._done(res) if ok else self._err(res)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_prog(self, msg, pct):
        self.root.after(0, lambda m=msg, p=pct: self._on_prog_main(m, p))

    def _on_prog_main(self, msg, pct):
        self._status.set(msg)
        if pct and pct > 0:
            self._progress.set(pct)
        self._append_log(msg)

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
        try:
            self.root.winfo_exists()
        except tk.TclError:
            return
        self.root.mainloop()


# =====================================================
# CLIモード
# =====================================================

def _run_cli(args):
    """コマンドライン引数で直接処理を実行する"""
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "耳コピ自動生成ツール v7.0 - Demucs×BasicPitch×FluidSynth\n"
            "本ツールは原曲音源を再配布しません。MIDI採譜後にサンプル音源で"
            "再合成した出力のみを生成します（原盤権保護）。"
        )
    )
    parser.add_argument("input", help="入力音楽ファイル（MP3/WAV/M4A/FLAC）")
    parser.add_argument("-o", "--output", help="出力MP3パス（省略時: 同じフォルダに 耳コピ_*.mp3）")
    parser.add_argument("--mode", choices=["ai", "ai_inst"],
                        default="ai",
                        help="処理モード: ai (推奨) / ai_inst (ガイドメロディなし)")
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

    engine = EarCopyEngine(mode=opts.mode)
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
# 追加モジュール: 特徴量抽出 / 後処理 / MIDI出力 / 分離推論
# =====================================================
# 既存 EarCopyEngine のメソッドと独立して外部からも利用可能な
# スタンドアロン関数・クラス群。既存コードは一切変更しない。

import logging as _logging

_mod_logger = _logging.getLogger("mimikopi.ext")


class FeatureExtractor:
    """Mel + CQT + Chroma を結合した特徴量を返す。

    既存 EarCopyEngine の個別解析（_detect_all_notes, _estimate_key 等）と
    独立して利用可能。モデル学習やリアルタイム推論の入力として使う想定。

    変更理由: ②特徴量強化 — 単一特徴量ではなく3種を結合して表現力を高める。
    CPU動作前提、GPU依存なし。
    """

    def __init__(self, sr=44100, hop_length=512,
                 n_mels=128, n_cqt_bins=84, n_chroma=12):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_cqt_bins = n_cqt_bins
        self.n_chroma = n_chroma

    @property
    def feature_dim(self):
        """結合後の特徴次元数 (n_mels + n_cqt_bins + n_chroma)"""
        return self.n_mels + self.n_cqt_bins + self.n_chroma

    def extract(self, y):
        """音声 ndarray → 結合特徴量 (T, feature_dim)

        Args:
            y: 1D float32 ndarray (モノラル音声)
        Returns:
            np.ndarray shape (T, feature_dim) — T はフレーム数
        """
        feats = []

        # --- Mel スペクトログラム ---
        try:
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sr, hop_length=self.hop_length,
                n_mels=self.n_mels, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            feats.append(mel_db)
        except Exception as e:
            _mod_logger.warning("Mel extraction failed: %s", e)
            feats.append(np.zeros((self.n_mels, 1)))

        # --- CQT (Constant-Q Transform) ---
        try:
            cqt = np.abs(librosa.cqt(
                y=y, sr=self.sr, hop_length=self.hop_length,
                n_bins=self.n_cqt_bins, bins_per_octave=12,
                fmin=librosa.note_to_hz('C1')))
            cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
            feats.append(cqt_db)
        except Exception as e:
            _mod_logger.warning("CQT extraction failed: %s", e)
            feats.append(np.zeros((self.n_cqt_bins, 1)))

        # --- Chroma (CQT ベース) ---
        try:
            chroma = librosa.feature.chroma_cqt(
                y=y, sr=self.sr, hop_length=self.hop_length,
                n_chroma=self.n_chroma)
            feats.append(chroma)
        except Exception as e:
            _mod_logger.warning("Chroma extraction failed: %s", e)
            feats.append(np.zeros((self.n_chroma, 1)))

        # フレーム数を最大に揃えて結合
        T = max(f.shape[1] for f in feats)
        aligned = []
        for f in feats:
            if f.shape[1] < T:
                f = np.pad(f, ((0, 0), (0, T - f.shape[1])), mode='edge')
            elif f.shape[1] > T:
                f = f[:, :T]
            aligned.append(f)
        combined = np.concatenate(aligned, axis=0)  # (feature_dim, T)
        return combined.T  # (T, feature_dim)


def postprocess_notes(notes, tempo=120.0, min_dur=0.04,
                      gap_fill=0.03, grid_snap=True):
    """ノートリストに標準後処理を適用するスタンドアロン関数。

    変更理由: ④後処理追加 — EarCopyEngine 内部の _refine_notes / _rebuild_rhythm
    を外部から単独で利用できるようラップ。

    Args:
        notes    : [(onset_s, dur_s, midi, vel), ...]
        tempo    : BPM
        min_dur  : これ未満のノートを削除（秒）
        gap_fill : 同ピッチ間ギャップをこれ以下なら結合（秒）
        grid_snap: 16分音符グリッドへの軽量スナップ
    Returns:
        [(onset_s, dur_s, midi, vel), ...]
    """
    if not notes:
        return notes
    _mod_logger.debug("postprocess_notes: input %d notes", len(notes))

    # 1. 短音削除
    notes = [(t, d, m, v) for t, d, m, v in notes if d >= min_dur]

    # 2. 同ピッチのギャップ補完（レガート接続）
    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n[2], []).append(list(n))
    result = []
    for midi, group in by_pitch.items():
        group = sorted(group, key=lambda x: x[0])
        for i in range(len(group) - 1):
            t, d, m, v = group[i]
            t_next = group[i + 1][0]
            gap = t_next - (t + d)
            if 0 < gap <= gap_fill:
                group[i] = [t, d + gap * 0.9, m, v]
        result.extend(tuple(g) for g in group)
    notes = sorted(result, key=lambda x: x[0])

    # 3. 16分音符グリッドスナップ（50ms以内のみ、40%強度）
    if grid_snap and tempo > 0:
        beat_dur = 60.0 / max(tempo, 40)
        grid_step = beat_dur / 4
        snapped = []
        for (t, d, m, v) in notes:
            nearest = round(t / grid_step) * grid_step
            if abs(nearest - t) <= 0.05:
                t = t + (nearest - t) * 0.4
            snapped.append((t, d, m, v))
        notes = snapped

    _mod_logger.debug("postprocess_notes: output %d notes", len(notes))
    return sorted(notes, key=lambda x: x[0])


def export_midi(notes, output_path, tempo=120.0, program=0, channel=0):
    """ノートリスト → MIDI ファイル出力。

    変更理由: ⑤MIDI出力改善 — pretty_midi で正確な note_on/off を処理。
    未インストール時は既存の mido でフォールバック。

    Args:
        notes       : [(onset_s, dur_s, midi, vel), ...]
        output_path : 保存先パス (.mid)
        tempo       : BPM
        program     : GM プログラム番号 (0=ピアノ)
        channel     : MIDI チャンネル (0-15)
    Returns:
        bool — 成功なら True
    """
    if not notes:
        _mod_logger.warning("export_midi: empty notes list")
        return False

    # pretty_midi を優先
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
        inst = pretty_midi.Instrument(program=program, name="melody")
        for (t, d, m, v) in notes:
            note = pretty_midi.Note(
                velocity=int(np.clip(v, 1, 127)),
                pitch=int(np.clip(m, 0, 127)),
                start=float(t),
                end=float(t + max(d, 0.01)),
            )
            inst.notes.append(note)
        pm.instruments.append(inst)
        pm.write(str(output_path))
        _mod_logger.info("export_midi (pretty_midi): %s", output_path)
        return True
    except ImportError:
        _mod_logger.info("pretty_midi 未インストール、mido にフォールバック")
    except Exception as e:
        _mod_logger.warning("pretty_midi export failed: %s — mido にフォールバック", e)

    # mido フォールバック（既存 _save_midi と同等ロジック）
    try:
        mid = MidiFile(type=0, ticks_per_beat=480)
        tpb = 480
        us = int(60_000_000 / max(tempo, 1))
        def s2t(s):
            return int(s * tempo / 60 * tpb)
        trk = MidiTrack()
        mid.tracks.append(trk)
        trk.append(MetaMessage('set_tempo', tempo=us, time=0))
        trk.append(Message('program_change', channel=channel,
                           program=program, time=0))
        evs = []
        for (t, d, m, v) in notes:
            note = int(np.clip(m, 0, 127))
            vel = int(np.clip(v, 1, 127))
            evs.append((s2t(t), Message(
                'note_on', channel=channel, note=note,
                velocity=vel, time=0)))
            evs.append((s2t(t + d), Message(
                'note_off', channel=channel, note=note,
                velocity=0, time=0)))
        evs.sort(key=lambda x: x[0])
        prev = 0
        for tick, msg in evs:
            msg.time = max(0, tick - prev)
            trk.append(msg)
            prev = tick
        mid.save(str(output_path))
        _mod_logger.info("export_midi (mido fallback): %s", output_path)
        return True
    except Exception as e:
        _mod_logger.error("export_midi failed: %s", e)
        return False


def infer_with_separation(input_path, output_midi_path=None,
                          mode="ai", on_progress=None):
    """音源分離 (Demucs) → other ステムのみ耳コピ → ノートリスト返却。

    変更理由: ①音源分離追加 — 既存 _demucs_separate をスタンドアロンで
    利用可能にし、other.wav のみを耳コピに使用する。

    Args:
        input_path      : 入力音声ファイルパス
        output_midi_path: MIDI 保存先 (None = 保存しない)
        mode            : EarCopyEngine のモード ("ai" 推奨)
        on_progress     : 進捗コールバック fn(msg, pct)
    Returns:
        [(onset_s, dur_s, midi, vel), ...] — other ステムの採譜結果
    """
    _mod_logger.info("infer_with_separation: %s", input_path)

    engine = EarCopyEngine(on_progress=on_progress, mode=mode)

    # ① 音源分離 (Demucs)
    try:
        if not _ensure_ai_packages(on_progress or (lambda m, p=None: None)):
            _mod_logger.warning("AI パッケージ準備失敗")
    except Exception:
        pass
    stems = engine._demucs_separate(input_path)

    if stems is None:
        _mod_logger.warning("Demucs 失敗、入力全体をフォールバック解析します")
        y, sr_use = engine._load(input_path)
        other_audio = y
    else:
        other_audio = stems.get("other")
        sr_use = SR
        if other_audio is None:
            _mod_logger.error("other ステムが見つかりません")
            return []

    # ② other ステムを前処理
    other_stem = engine._preprocess_stem(other_audio, sr_use, 'other')

    # ③ 採譜 (Basic Pitch → fallback: CQT+pyin)
    notes = engine._basic_pitch_notes(other_stem, sr_use, is_vocal=False)
    notes = engine._transformer_refine(notes)
    notes = engine._remove_overlapping_notes(notes)
    _mod_logger.info("  採譜結果: %d ノート", len(notes))

    # ④ テンポ推定 + 後処理
    try:
        tempo, _beats = engine._tempo(other_audio, sr_use)
        tempo = float(tempo) if tempo else 120.0
    except Exception:
        tempo = 120.0
    notes = postprocess_notes(notes, tempo=tempo)

    # ④-b 音楽理論補正
    try:
        from music_theory import correct_and_humanize as _mt_correct
        beat_arr = _beats if '_beats' in dir() and _beats is not None else None
        notes = _mt_correct(notes, audio=other_audio, sr=sr_use,
                            tempo=tempo, beat_times=beat_arr)
        _mod_logger.info("  音楽理論補正後: %d ノート", len(notes))
    except ImportError:
        pass
    except Exception as e:
        _mod_logger.debug("  音楽理論補正スキップ: %s", e)

    # ⑤ MIDI 保存 (オプション)
    if output_midi_path:
        export_midi(notes, output_midi_path, tempo=tempo)
        _mod_logger.info("  MIDI 保存: %s", output_midi_path)

    # 原盤権保護: 採譜が完了したので Demucs 由来の原音ステムを破棄する
    try:
        if stems is not None:
            stems.clear()
    except Exception:
        pass

    return notes


# =====================================================
# 軽量耳コピモデル (LightTransformerModel)
# =====================================================
# 変更理由: MusicTransformer はノートトークン精製用。こちらは
# 音声特徴量 → フレームレベルノート検出の専用軽量推論モデル。
# CPU 前提: ~940K パラメータ、推論約 10ms/0.5sec。
#
# 既存コードは一切変更しない。新クラス・関数のみ追加。

# MIDI ピッチクラス定義: クラス 0 = MIDI 36 (C2), クラス 59 = MIDI 95 (B6)
_PITCH_OFFSET = 36
_NUM_PITCHES  = 60


def _decode_predictions(pitch_prob, onset_prob, frame_sec,
                         pitch_thr=0.4, onset_thr=0.5):
    """フレーム確率行列 → ノートイベントリスト。

    純粋 numpy のみ（torch 依存なし）。ステートマシンで
    onset + pitch アクティブ区間を追跡してノートに変換する。

    Args:
        pitch_prob : (T, _NUM_PITCHES) float ndarray 0-1
        onset_prob : (T, _NUM_PITCHES) float ndarray 0-1
        frame_sec  : 1 フレームの秒数 (hop_length / sr)
    Returns:
        [(onset_s, dur_s, midi, vel), ...]
    """
    T, P = pitch_prob.shape
    notes = []

    for p in range(P):
        midi_note = p + _PITCH_OFFSET
        active = False
        start_t = 0
        vel_acc = []

        for t in range(T):
            p_on = pitch_prob[t, p] >= pitch_thr
            o_on = onset_prob[t, p] >= onset_thr

            if not active:
                if p_on and o_on:          # onset → ノート開始
                    active = True
                    start_t = t
                    vel_acc = [pitch_prob[t, p]]
            else:
                if p_on and not o_on:      # 持続中
                    vel_acc.append(pitch_prob[t, p])
                else:
                    # 音が消えるか再 onset → 現ノート終了
                    dur = (t - start_t) * frame_sec
                    if dur >= 0.02 and vel_acc:
                        vel = int(np.clip(
                            40 + float(np.mean(vel_acc)) * 80, 30, 120))
                        notes.append((start_t * frame_sec, dur, midi_note, vel))
                    active = False
                    vel_acc = []
                    if p_on and o_on:      # 即座に新 onset
                        active = True
                        start_t = t
                        vel_acc = [pitch_prob[t, p]]

        # ループ末尾で発音中のノートを閉じる
        if active and vel_acc:
            dur = (T - start_t) * frame_sec
            if dur >= 0.02:
                vel = int(np.clip(
                    40 + float(np.mean(vel_acc)) * 80, 30, 120))
                notes.append((start_t * frame_sec, dur, midi_note, vel))

    return sorted(notes, key=lambda x: x[0])


if HAS_TORCH:
    class LightCNN(nn.Module):
        """軽量 2 層 Conv1d: 特徴量を Transformer 入力次元へ圧縮する。

        変更理由: 224 次元特徴量を Transformer に直接渡すと重いため、
        CNN で 128 次元に圧縮してから渡す。
        BatchNorm は batch_size=1 で不安定になるため省略。

        入力: (B, T, in_channels)
        出力: (B, T, d_model)
        """

        def __init__(self, in_channels=224, d_model=128, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.conv1 = nn.Conv1d(in_channels, d_model,
                                   kernel_size, padding=pad)
            self.conv2 = nn.Conv1d(d_model, d_model,
                                   kernel_size, padding=pad)
            self.relu = nn.ReLU()

        def forward(self, x):
            # (B, T, C) → (B, C, T) → Conv → (B, d_model, T) → (B, T, d_model)
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            return x.transpose(1, 2)

    class LightTransformerModel(nn.Module):
        """軽量耳コピモデル: LightCNN + 2 層 Transformer + pitch/onset ヘッド。

        設計思想:
          - CPU 前提設計 (~940K パラメータ)
          - MusicTransformer (トークン精製用 4 層) とは用途が異なる
          - フレーム単位の pitch/onset を同時推定する二頭蛇構造
          - ⑨ 0.5 秒フレーム入力前提、将来的なストリーム処理にも対応

        パラメータ数概算:
          CNN 2 層: ~270K, Transformer 2 層 d=128: ~656K,
          ヘッド 2 本: ~15K → 合計 ~940K
        """

        def __init__(self, feature_dim=224, d_model=128, nhead=4,
                     num_layers=2, dim_ff=256,
                     num_pitches=_NUM_PITCHES, dropout=0.1,
                     max_len=512):
            super().__init__()
            self.d_model = d_model
            self.num_pitches = num_pitches
            self._max_len = max_len
            # ① CNN 特徴圧縮 (feature_dim → d_model)
            self.cnn = LightCNN(feature_dim, d_model)
            # ② 学習済み位置エンコーディング
            self.pos_enc = nn.Embedding(max_len, d_model)
            # ③ 軽量 2 層 Transformer (CPU 動作を最優先)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_ff, dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                enc_layer, num_layers=num_layers)
            # ④ 出力ヘッド: pitch (アクティブ音) + onset (発音タイミング)
            self.pitch_head = nn.Linear(d_model, num_pitches)
            self.onset_head = nn.Linear(d_model, num_pitches)

        def forward(self, x):
            """x: (B, T, feature_dim)
            戻り値: pitch_logits, onset_logits — 各 (B, T, num_pitches)
            """
            B, T, _ = x.shape
            h = self.cnn(x)                              # (B, T, d_model)
            T_enc = min(T, self._max_len)
            pos = torch.arange(T_enc, device=x.device).unsqueeze(0)
            h = h[:, :T_enc] + self.pos_enc(pos)        # 位置情報付加
            h = self.transformer(h)                      # (B, T, d_model)
            return self.pitch_head(h), self.onset_head(h)

        @torch.no_grad()
        def predict(self, features_np, top_k=5,
                    pitch_thr=0.4, onset_thr=0.5,
                    sr=44100, hop_length=512):
            """numpy 特徴量 → ノートリスト（推論専用）。

            変更理由: ⑦推論最適化 —
              - torch.no_grad() でメモリ/計算を削減
              - バッチサイズ 1 固定
              - numpy 変換を予測後の 1 回のみに限定

            Args:
                features_np : (T, feature_dim) ndarray
                top_k       : ⑧ 各フレームで採用する最大ノート数
            Returns:
                [(onset_s, dur_s, midi, vel), ...]
            """
            self.eval()
            # バッチサイズ 1 固定 (B=1)
            feat = torch.from_numpy(
                features_np.astype(np.float32)
            ).unsqueeze(0)                              # (1, T, feature_dim)
            T = feat.shape[1]
            if T > self._max_len:
                feat = feat[:, :self._max_len, :]
                T = self._max_len

            pitch_logits, onset_logits = self(feat)
            pitch_prob = torch.sigmoid(pitch_logits[0])  # (T, num_pitches)
            onset_prob = torch.sigmoid(onset_logits[0])  # (T, num_pitches)

            # ⑧ Top-K フィルタリング: 各フレームで上位 K 音のみ採用
            if 0 < top_k < self.num_pitches:
                _, topk_idx = pitch_prob.topk(top_k, dim=-1)
                mask = torch.zeros_like(pitch_prob, dtype=torch.bool)
                mask.scatter_(-1, topk_idx, True)
                pitch_prob = pitch_prob * mask

            # numpy 変換は 1 回のみ（推論最適化）
            p_np = pitch_prob.cpu().numpy()
            o_np = onset_prob.cpu().numpy()

            # ⑧ 時間方向スムージング (EMA でフレームブレを低減)
            try:
                from music_theory import smooth_predictions
                p_np, o_np = smooth_predictions(p_np, o_np,
                                                method='ema', alpha=0.45)
            except ImportError:
                pass

            frame_sec = hop_length / sr

            return _decode_predictions(
                p_np, o_np, frame_sec,
                pitch_thr=pitch_thr, onset_thr=onset_thr,
            )

    class CombinedLoss(nn.Module):
        """BCE(pitch) + 2×BCE(onset) + 時間方向スムージング Loss。

        変更理由: ⑤損失改善 —
          - onset を 2 倍重視: 音の立ち上がりタイミング精度を向上
          - スムージング Loss: 隣接フレーム差を抑えてチラつきを低減
            計算式: mean( (logit[t+1] - logit[t])^2 )
        """

        def __init__(self, onset_weight=2.0, smooth_weight=0.1):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.onset_weight = onset_weight
            self.smooth_weight = smooth_weight

        def forward(self, pitch_logits, onset_logits,
                    pitch_target, onset_target):
            """
            *_logits : (B, T, num_pitches) — 生ロジット
            *_target : (B, T, num_pitches) — 0/1 float テンソル
            戻り値: (total_loss, {"pitch": float, "onset": float, "smooth": float})
            """
            loss_pitch = self.bce(pitch_logits, pitch_target)
            loss_onset = (self.bce(onset_logits, onset_target)
                          * self.onset_weight)
            # 時間方向スムージング: 隣接フレームの差の 2 乗平均
            smooth = ((pitch_logits[:, 1:] - pitch_logits[:, :-1]) ** 2).mean()
            total = loss_pitch + loss_onset + smooth * self.smooth_weight
            return total, {
                "pitch":  float(loss_pitch),
                "onset":  float(loss_onset),
                "smooth": float(smooth),
            }

else:
    # torch 未インストール時のスタブ（既存動作に影響なし）
    class LightCNN:            # type: ignore[no-redef]
        def __init__(self, **_): pass

    class LightTransformerModel:   # type: ignore[no-redef]
        def __init__(self, **_): pass
        def predict(self, *_, **__): return []

    class CombinedLoss:            # type: ignore[no-redef]
        def __init__(self, **_): pass
        def __call__(self, *_, **__): return 0.0, {}


def infer_with_light_model(audio, sr=44100, hop_length=512,
                            model=None, top_k=5,
                            pitch_thr=0.4, onset_thr=0.5,
                            tempo=120.0):
    """音声 ndarray → 軽量モデルでノートリストを推論する。

    変更理由: FeatureExtractor + LightTransformerModel + postprocess_notes を
    一括して呼び出すエンドツーエンド推論関数。
    torch 未インストール時は空リストを返す（後処理は既存コードに委譲可能）。

    Args:
        audio     : 1D float32 ndarray（モノラル）
        sr        : サンプリングレート
        hop_length: CQT/Mel ホップ長
        model     : LightTransformerModel インスタンス (None=新規作成)
        top_k     : フレームごとの最大採用ノート数
        pitch_thr : ピッチ検出閾値
        onset_thr : オンセット検出閾値
        tempo     : BPM（後処理グリッド量子化用）
    Returns:
        [(onset_s, dur_s, midi, vel), ...]
    """
    if not HAS_TORCH:
        _mod_logger.warning("infer_with_light_model: torch 未インストール → 空リスト")
        return []

    # ① 特徴量抽出 (Mel + CQT + Chroma)
    fe = FeatureExtractor(sr=sr, hop_length=hop_length)
    features = fe.extract(audio)                     # (T, 224)

    # ② モデル推論
    if model is None:
        model = LightTransformerModel(feature_dim=fe.feature_dim)
    notes = model.predict(
        features, top_k=top_k,
        pitch_thr=pitch_thr, onset_thr=onset_thr,
        sr=sr, hop_length=hop_length,
    )

    # ③ 後処理（短音削除・ギャップ補完・グリッド量子化）
    notes = postprocess_notes(notes, tempo=tempo)

    # ④ 音楽理論補正
    try:
        from music_theory import correct_and_humanize as _mt_correct
        notes = _mt_correct(notes, audio=audio, sr=sr, tempo=tempo)
    except ImportError:
        pass
    except Exception:
        pass

    _mod_logger.info("infer_with_light_model: %d notes", len(notes))
    return notes


def benchmark_light_model(duration_sec=0.5, sr=44100, n_runs=5):
    """LightTransformerModel の CPU 推論時間・メモリ使用量を計測する。

    ⑨ 軽量化チェック: 推論時間 / メモリ / CPU 動作確認用。
    GPU が存在しても CPU のみで動作することを確認する。

    Args:
        duration_sec : ベンチマーク用音声の長さ（秒）
        sr           : サンプリングレート
        n_runs       : 計測反復回数（平均を算出）
    Returns:
        dict: {
            "mean_ms"   : 平均推論時間（ミリ秒）,
            "min_ms"    : 最短推論時間（ミリ秒）,
            "max_ms"    : 最長推論時間（ミリ秒）,
            "peak_mb"   : ピークメモリ使用量（MB）,
            "cpu_only"  : True = CPU のみで動作,
            "has_torch" : torch インストール済みかどうか,
            "params"    : モデルパラメータ数,
        }
    """
    import time
    import tracemalloc

    result = {
        "mean_ms":  None,
        "min_ms":   None,
        "max_ms":   None,
        "peak_mb":  None,
        "cpu_only": True,
        "has_torch": HAS_TORCH,
        "params":   0,
    }

    if not HAS_TORCH:
        _mod_logger.warning("benchmark_light_model: torch 未インストール — スキップ")
        return result

    import torch

    # ダミー音声 (ランダムノイズ 0.5秒)
    audio = np.random.randn(int(duration_sec * sr)).astype(np.float32) * 0.1

    # 特徴量抽出
    fe = FeatureExtractor(sr=sr)
    features = fe.extract(audio)          # (T, 224)

    # モデル生成 (CPU 固定)
    model = LightTransformerModel(feature_dim=fe.feature_dim)
    model.eval()

    # パラメータ数
    result["params"] = sum(p.numel() for p in model.parameters())

    # CPU 動作確認
    result["cpu_only"] = not next(model.parameters()).is_cuda

    # ウォームアップ (JIT キャッシュ等を安定させる)
    _ = model.predict(features)

    # 推論時間計測
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(features)
        times.append(time.perf_counter() - t0)

    result["mean_ms"] = float(np.mean(times) * 1000)
    result["min_ms"]  = float(np.min(times)  * 1000)
    result["max_ms"]  = float(np.max(times)  * 1000)

    # ピークメモリ計測
    tracemalloc.start()
    _ = model.predict(features)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_mb"] = peak / 1e6

    _mod_logger.info(
        "benchmark_light_model: %.1f ms (mean), %.1f MB peak, CPU=%s, params=%d",
        result["mean_ms"], result["peak_mb"],
        result["cpu_only"], result["params"],
    )
    return result


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
