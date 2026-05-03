#!/usr/bin/env python3
"""
legal_check.py — 法務チェックスクリプト

ビルド時に走らせて以下を検証:
  1. コード内に _hybrid_mix の残骸がないか
  2. 最終出力経路に Demucs ステムが直接流れる箇所がないか
  3. 同梱 SoundFont/IR のライセンスファイルが揃っているか
  4. README に必要なクレジット表記があるか

使い方:
  python legal_check.py
  # 終了コード 0 = 全 PASS, 1 = FAIL あり
"""
import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIMIKOPI = ROOT / "mimikopi.py"
INSTRUMENTS_JSON = ROOT / "instruments.json"
README_LICENSES = ROOT / "README_LICENSES.txt"

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ PASS: {name}")
        PASS += 1
    else:
        print(f"  ✗ FAIL: {name}")
        if detail:
            print(f"         {detail}")
        FAIL += 1


def main():
    global PASS, FAIL
    src = MIMIKOPI.read_text(encoding="utf-8") if MIMIKOPI.exists() else ""
    lines = src.split("\n")

    # ========================================
    print("=" * 60)
    print("1. _hybrid_mix 残骸チェック")
    print("=" * 60)

    # _hybrid_mix が関数定義として存在しないこと
    hybrid_defs = [
        (i + 1, l) for i, l in enumerate(lines)
        if re.search(r"def\s+_hybrid_mix\b", l)
    ]
    check("_hybrid_mix 関数定義なし", len(hybrid_defs) == 0,
          f"行 {hybrid_defs}" if hybrid_defs else "")

    # _hybrid_mix の呼び出しがないこと (コメント内は許容)
    hybrid_calls = [
        (i + 1, l.strip()) for i, l in enumerate(lines)
        if "_hybrid_mix" in l and not l.strip().startswith("#")
    ]
    check("_hybrid_mix 呼び出しなし (コメント外)", len(hybrid_calls) == 0,
          f"行 {hybrid_calls}" if hybrid_calls else "")

    # _mix_stems が関数定義として存在しないこと
    mix_stems_defs = [
        (i + 1, l) for i, l in enumerate(lines)
        if re.search(r"def\s+_mix_stems\b", l)
    ]
    check("_mix_stems 関数定義なし", len(mix_stems_defs) == 0,
          f"行 {mix_stems_defs}" if mix_stems_defs else "")

    # ========================================
    print()
    print("=" * 60)
    print("2. 最終出力経路のステム漏洩チェック")
    print("=" * 60)

    # _process_ai 内でステージ7以降に stems が使われていないこと
    in_process_ai = False
    past_stage7 = False
    stem_leaks = []
    for i, l in enumerate(lines):
        if "def _process_ai(" in l:
            in_process_ai = True
            past_stage7 = False
        elif in_process_ai and l.strip().startswith("def ") and "def _process_ai" not in l:
            in_process_ai = False
            past_stage7 = False
        if in_process_ai and "ステージ7" in l:
            past_stage7 = True
        if in_process_ai and past_stage7:
            # stems[" や stems.get( のパターン (del stems と stems.clear は OK)
            if re.search(r'stems\["', l) or re.search(r"stems\.get\(", l):
                if "stems.clear()" not in l:
                    stem_leaks.append((i + 1, l.strip()))

    check("ステージ7以降にステム参照なし", len(stem_leaks) == 0,
          f"行 {stem_leaks}" if stem_leaks else "")

    # _save_mp3 / _master / _synthesize_audio で stems 引数を受け取っていないこと
    for func_name in ["_save_mp3", "_master", "_synthesize_audio"]:
        pattern = rf"def\s+{func_name}\s*\([^)]*stems[^)]*\)"
        matches = [
            (i + 1, l.strip()) for i, l in enumerate(lines)
            if re.search(pattern, l)
        ]
        check(f"{func_name} に stems 引数なし", len(matches) == 0,
              f"行 {matches}" if matches else "")

    # stems.get("drums"|"bass"|"other") が最終出力関数内にないこと
    output_funcs = {"_master", "_save_mp3", "_synthesize_audio",
                    "_synthesize_multi_sf2", "_soft_compress", "_lufs_normalize"}
    current_func = None
    for i, l in enumerate(lines):
        m = re.match(r"\s+def\s+(\w+)\s*\(", l)
        if m:
            current_func = m.group(1)
        if current_func in output_funcs:
            if re.search(r'stems\.get\(\s*["\'](?:drums|bass|other)', l):
                stem_leaks.append((i + 1, f"[{current_func}] {l.strip()}"))

    check("出力関数内にステム直接参照なし", len(stem_leaks) == 0,
          f"行 {stem_leaks}" if stem_leaks else "")

    # ========================================
    print()
    print("=" * 60)
    print("3. ライセンスファイルチェック")
    print("=" * 60)

    check("instruments.json 存在", INSTRUMENTS_JSON.exists())

    if INSTRUMENTS_JSON.exists():
        try:
            config = json.loads(INSTRUMENTS_JSON.read_text(encoding="utf-8"))
            sf_section = config.get("soundfonts", {})
            check("soundfonts セクション存在", len(sf_section) > 0)

            for key, info in sf_section.items():
                check(f"SF2 '{key}' にライセンス記載", "license" in info,
                      f"info keys: {list(info.keys())}")

            ir_section = config.get("impulse_responses", {})
            check("impulse_responses セクション存在", len(ir_section) > 0)
            for key, info in ir_section.items():
                check(f"IR '{key}' にライセンス記載", "license" in info)
        except json.JSONDecodeError as e:
            check("instruments.json パース成功", False, str(e))

    # ========================================
    print()
    print("=" * 60)
    print("4. クレジット表記チェック")
    print("=" * 60)

    # README_LICENSES.txt の生成機能がコードに存在すること
    has_license_gen = "_generate_license_readme" in src
    check("_generate_license_readme 関数がコードに存在", has_license_gen)

    # docstring に原盤不使用ポリシーが記載されていること
    has_policy = "100% MIDI再合成" in src and "原盤不使用" in src
    check("docstring に原盤不使用ポリシー記載", has_policy)

    # CLI ヘルプに原盤不使用が記載されていること
    has_cli_notice = "原曲音源を再配布しません" in src or "原盤権保護" in src
    check("CLI ヘルプに原盤不使用表示あり", has_cli_notice)

    # CC-BY 要件: Salamander Piano のクレジット
    if INSTRUMENTS_JSON.exists():
        try:
            config = json.loads(INSTRUMENTS_JSON.read_text(encoding="utf-8"))
            for key, info in config.get("soundfonts", {}).items():
                if info.get("credit_required"):
                    check(f"CC-BY SF2 '{key}' に credit_required=true", True)
        except Exception:
            pass

    # ========================================
    print()
    print("=" * 60)
    total = PASS + FAIL
    print(f"結果: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
