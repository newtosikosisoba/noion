@echo off
chcp 65001 >nul 2>&1
title Noion - AI 耳コピサービス

cd /d "%~dp0"

echo ============================================
echo   Noion - AI 耳コピ ^& 伴奏生成サービス
echo ============================================
echo.

:: Python チェック
where python >nul 2>&1
if errorlevel 1 (
    echo [エラー] Python が見つかりません。
    echo https://www.python.org/downloads/ からインストールしてください。
    pause
    exit /b 1
)

:: Node.js チェック (フロントエンドビルド用)
where node >nul 2>&1
if errorlevel 1 (
    echo [警告] Node.js が見つかりません。フロントエンドなしで起動します。
    echo API のみ: http://localhost:8000/docs
    goto :skip_frontend
)

:: Python 依存インストール
echo [1/4] Python 依存パッケージを確認中...
pip install -q -r requirements.txt 2>nul
pip install -q -r requirements-web.txt 2>nul
pip install -q fastapi uvicorn[standard] python-multipart aiofiles 2>nul

:: フロントエンド ビルド
if not exist "frontend\dist\index.html" (
    echo [2/4] フロントエンドをビルド中...
    cd frontend
    if not exist "node_modules" (
        call npm install
    )
    call npm run build
    cd ..
    if not exist "frontend\dist\index.html" (
        echo [警告] フロントエンドのビルドに失敗しました。API のみで起動します。
    )
) else (
    echo [2/4] フロントエンド ビルド済み — スキップ
)
goto :start_server

:skip_frontend
echo [1/4] Python 依存パッケージを確認中...
pip install -q -r requirements.txt 2>nul
pip install -q -r requirements-web.txt 2>nul
pip install -q fastapi uvicorn[standard] python-multipart aiofiles 2>nul
echo [2/4] フロントエンド — スキップ

:start_server
echo [3/4] サーバーを起動中...
echo.
echo   URL: http://localhost:8000
echo   API: http://localhost:8000/docs
echo   終了: Ctrl+C または このウィンドウを閉じる
echo.

:: 3秒後にブラウザを開く (バックグラウンド)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

echo [4/4] 起動完了！ブラウザが自動で開きます。
echo ============================================
echo.

python run.py

echo.
echo サーバーが停止しました。
pause
