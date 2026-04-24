#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""api.py — FastAPI Web API for 耳コピ自動生成ツール

変更理由: ⑧Web API追加 — 音源アップロード → MIDI 生成を REST で提供。
既存 mimikopi.py を壊さず、独立モジュールとして新規作成。

依存: fastapi, uvicorn, python-multipart
起動: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("api")

# FastAPI インポート（未インストール時は起動エラーメッセージ表示）
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:
    print("FastAPI が未インストールです:")
    print("  pip install fastapi uvicorn[standard] python-multipart")
    sys.exit(1)

# mimikopi から必要な関数・クラスをインポート
sys.path.insert(0, os.path.dirname(__file__))
try:
    from mimikopi import (
        EarCopyEngine, infer_with_separation,
        postprocess_notes, export_midi, SR,
    )
except ImportError as e:
    print(f"mimikopi.py のインポートに失敗: {e}")
    sys.exit(1)


# =====================================================
# FastAPI アプリケーション
# =====================================================

app = FastAPI(
    title="mimikopi API",
    description="音源ファイルをアップロードして MIDI に変換する耳コピ API",
    version="1.0.0",
)


@app.get("/health")
def health():
    """ヘルスチェック"""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="音声ファイル (MP3, WAV 等)"),
    mode: str = "ai",
    output_format: str = "midi",
):
    """音源ファイルをアップロードし、耳コピ → MIDI を生成して返す。

    Args:
        file          : アップロード音声ファイル
        mode          : 処理モード ("ai", "ai_inst")
        output_format : 出力形式 ("midi" のみ)
    Returns:
        MIDI ファイル (application/octet-stream)
    """
    # バリデーション
    if mode not in ("ai", "ai_inst"):
        raise HTTPException(400, f"未対応モード: {mode}")
    if output_format != "midi":
        raise HTTPException(400, f"未対応出力形式: {output_format}")

    # 一時ファイルに保存
    suffix = Path(file.filename).suffix if file.filename else ".mp3"
    if suffix not in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"):
        raise HTTPException(400, f"未対応ファイル形式: {suffix}")

    tmp_input = None
    tmp_midi = None
    try:
        # 入力ファイルを一時保存
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="mimikopi_api_"
        ) as f:
            tmp_input = f.name
            content = await file.read()
            f.write(content)

        log.info("transcribe: %s (%d bytes, mode=%s)",
                 file.filename, len(content), mode)

        # MIDI 出力先
        tmp_midi = tmp_input + ".mid"

        # 採譜実行（infer_with_separation を利用）
        progress_msgs = []
        def on_progress(msg, pct=None):
            progress_msgs.append(msg)
            log.info("  [%s%%] %s", pct if pct else "--", msg)

        notes = infer_with_separation(
            tmp_input,
            output_midi_path=tmp_midi,
            mode=mode,
            on_progress=on_progress,
        )

        if not notes:
            raise HTTPException(500, "採譜結果が空です")

        if not Path(tmp_midi).exists():
            raise HTTPException(500, "MIDI ファイルの生成に失敗しました")

        log.info("transcribe complete: %d notes → %s", len(notes), tmp_midi)

        # MIDI ファイルを返却
        return FileResponse(
            path=tmp_midi,
            media_type="audio/midi",
            filename=Path(file.filename).stem + ".mid"
            if file.filename else "output.mid",
            background=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("transcribe error: %s", e, exc_info=True)
        raise HTTPException(500, f"処理中にエラーが発生: {str(e)}")
    finally:
        # 入力ファイルのクリーンアップ
        if tmp_input and os.path.exists(tmp_input):
            try:
                os.unlink(tmp_input)
            except Exception:
                pass


@app.post("/transcribe/notes")
async def transcribe_notes(
    file: UploadFile = File(..., description="音声ファイル (MP3, WAV 等)"),
    mode: str = "ai",
):
    """音源ファイルをアップロードし、ノートリストを JSON で返す。

    MIDI ファイルではなく構造化データが必要な場合に使用。
    """
    if mode not in ("ai", "ai_inst"):
        raise HTTPException(400, f"未対応モード: {mode}")

    suffix = Path(file.filename).suffix if file.filename else ".mp3"
    if suffix not in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"):
        raise HTTPException(400, f"未対応ファイル形式: {suffix}")

    tmp_input = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="mimikopi_api_"
        ) as f:
            tmp_input = f.name
            content = await file.read()
            f.write(content)

        notes = infer_with_separation(
            tmp_input, mode=mode,
            on_progress=lambda m, p=None: None,
        )

        return JSONResponse({
            "notes": [
                {"onset": round(t, 4), "duration": round(d, 4),
                 "midi": m, "velocity": v}
                for t, d, m, v in notes
            ],
            "count": len(notes),
        })

    except Exception as e:
        log.error("transcribe_notes error: %s", e, exc_info=True)
        raise HTTPException(500, f"処理中にエラーが発生: {str(e)}")
    finally:
        if tmp_input and os.path.exists(tmp_input):
            try:
                os.unlink(tmp_input)
            except Exception:
                pass


# =====================================================
# 直接実行時は uvicorn で起動
# =====================================================

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("uvicorn が未インストールです: pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
