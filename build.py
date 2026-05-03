#!/usr/bin/env python3
"""
build.py — PyInstaller ビルドスクリプト

Windows .exe / macOS .app をビルドする。
SoundFont は配布物に含めず、初回起動時にダウンロードする方式。

使い方:
  python build.py              # 現在のプラットフォーム向けにビルド
  python build.py --onefile    # 単一ファイル .exe / .app 生成
  python build.py --check      # PyInstaller 利用可否チェックのみ
"""
import sys
import subprocess
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "mimikopi.py"
ICON_WIN = ROOT / "_assets" / "icon.ico"
ICON_MAC = ROOT / "_assets" / "icon.icns"
INSTRUMENTS_JSON = ROOT / "instruments.json"
MUSIC_THEORY = ROOT / "music_theory.py"

APP_NAME = "Noion"
VERSION = "1.0.0"


def check_pyinstaller():
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} 検出")
        return True
    except ImportError:
        print("✗ PyInstaller が見つかりません")
        print("  インストール: pip install pyinstaller")
        return False


def build(onefile=False):
    if not check_pyinstaller():
        return False

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # GUI アプリ (コンソール非表示)
    if platform.system() == "Windows":
        cmd.append("--noconsole")
        if ICON_WIN.exists():
            cmd.extend(["--icon", str(ICON_WIN)])
    elif platform.system() == "Darwin":
        cmd.append("--windowed")
        if ICON_MAC.exists():
            cmd.extend(["--icon", str(ICON_MAC)])

    # 追加データファイル
    sep = ";" if platform.system() == "Windows" else ":"
    if INSTRUMENTS_JSON.exists():
        cmd.extend(["--add-data", f"{INSTRUMENTS_JSON}{sep}."])
    if MUSIC_THEORY.exists():
        cmd.extend(["--add-data", f"{MUSIC_THEORY}{sep}."])

    # Hidden imports (動的インポートされるモジュール)
    hidden = [
        "numpy", "librosa", "soundfile", "scipy",
        "pydub", "mido",
        "music_theory",
    ]
    for h in hidden:
        cmd.extend(["--hidden-import", h])

    # AI 系パッケージは含めない（初回起動時にインストール）
    excludes = [
        "torch", "demucs", "basic_pitch", "tensorflow",
        "matplotlib", "IPython", "notebook",
    ]
    for e in excludes:
        cmd.extend(["--exclude-module", e])

    cmd.append(str(MAIN_SCRIPT))

    print(f"\nビルドコマンド:")
    print(f"  {' '.join(cmd)}")
    print(f"\nビルド開始...")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode == 0:
        dist_dir = ROOT / "dist"
        print(f"\n✓ ビルド完了: {dist_dir}")
        if onefile:
            if platform.system() == "Windows":
                print(f"  実行ファイル: {dist_dir / f'{APP_NAME}.exe'}")
            elif platform.system() == "Darwin":
                print(f"  アプリ: {dist_dir / f'{APP_NAME}.app'}")
            else:
                print(f"  実行ファイル: {dist_dir / APP_NAME}")
        else:
            print(f"  フォルダ: {dist_dir / APP_NAME}")
        return True
    else:
        print(f"\n✗ ビルド失敗 (exit code {result.returncode})")
        return False


def main():
    if "--check" in sys.argv:
        ok = check_pyinstaller()
        sys.exit(0 if ok else 1)

    onefile = "--onefile" in sys.argv
    ok = build(onefile=onefile)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
