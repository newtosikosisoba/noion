# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Noion is an AI ear-copy (耳コピ / automatic transcription) and accompaniment generation tool. It transcribes audio to MIDI using Demucs (stem separation) + Basic Pitch (polyphonic transcription), then resynthesizes a fully original accompaniment with FluidSynth + SoundFont. Output is tuned for "歌ってみた" (vocal cover) use: -16 LUFS / -1.5 dBTP with vocal-friendly stereo placement (see README.md).

The codebase, commit messages, docs, and most comments are in **Japanese**. Follow that convention (e.g. commit style: `feat: 〜`, `fix: 〜`).

## Legal Constraint (critical)

The final WAV/MP3 must be **100% MIDI resynthesis**. Demucs-separated stems are used only as transcription input and for scoring — they must never be mixed into the output. `python legal_check.py` enforces this (no `_hybrid_mix` remnants, no stem paths feeding the output, license files present) and must exit 0. Do not reintroduce any hybrid-mix path.

## Commands

```bash
# Dependencies (mimikopi.py also auto-installs core packages on first run;
# heavy AI packages — torch/demucs/basic-pitch — are lazily installed at process() time)
pip install -r requirements.txt -r requirements-web.txt

# Tests (run from repo root; AI packages are stubbed, so no torch/demucs needed)
pytest tests/
pytest tests/test_transcription.py -k chord        # single test

# Desktop tool
python mimikopi.py                                  # Tkinter GUI
python mimikopi.py input.mp3 -o out.mp3 --mode ai   # CLI (modes: ai, ai_inst)

# Web service (FastAPI on :8000, serves frontend/dist at / if built)
python run.py

# Frontend (Vite dev server on :5173 — CORS is preconfigured for this origin)
cd frontend && npm install && npm run dev
cd frontend && npm run build                        # tsc + vite build → frontend/dist

# Quality & verification
python scripts/measure_quality.py input.mp3 output.mp3 [--midi out.mid]
python benchmark.py                                 # process fixtures/, HTML A/B report in output/
python legal_check.py                               # legal gate (exit 0 = PASS)
python build.py [--onefile]                         # PyInstaller desktop build
```

`start.bat` is the Windows one-click launcher (installs deps, builds frontend, runs `run.py`). Keep it ASCII-only — non-ASCII characters have caused it to exit immediately.

## Architecture

### Core engine: `mimikopi.py` (single ~6700-line module)
Organized into `# =====` banner sections: auto-install, ffmpeg/SoundFont/FluidSynth auto-download & path caching (`.ffmpeg_path.json` etc.), `EarCopyEngine` (the pipeline), Tkinter `App` GUI, CLI entry, and standalone helpers (`infer_with_separation`, `postprocess_notes`, `export_midi`).

`EarCopyEngine.process()` pipeline: Demucs stem separation → Basic Pitch transcription per stem → note refinement (overlap removal, scale snap, chord filtering, rhythm rebuild, iterative refine scored against the original) → velocity floor + humanize → MIDI export → FluidSynth synthesis per part → post-processing (transient shaper, harmonic exciter, part EQ) → stereo positioning (per-part panning, spectral drum pan) → mastering (6-band parametric EQ, Mid/Side EQ, LUFS -16, true peak -1.5 dBTP, LR balance).

**Keep AI-package imports lazy** (inside functions): the test suite stubs `torch`/`demucs`/`basic_pitch`/`torchaudio` via `sys.modules` before importing mimikopi, so a top-level import of these breaks all tests.

### Satellite modules (pattern: extend without touching mimikopi.py)
New features are added as independent modules that operate on note lists or wrap the engine:
- `music_theory.py` — theory correction layer (chord/key estimation, scale filtering, quantization, harmony prioritization); entry point `correct_notes()`.
- `realtime.py` — mic input → realtime F0 → MIDI out (sounddevice + rtmidi).
- `api.py` — older minimal standalone REST API (upload → MIDI). The full web service lives in `webapp/`; don't confuse the two.

### Web service: `webapp/` + `frontend/`
- `webapp/main.py` — FastAPI app: routers (`health`, `auth`, `jobs`, `stripe`), rate-limit middleware, serves `frontend/dist` as static files.
- Auth: JWT (access/refresh) in `auth.py`; Stripe subscriptions in `services/stripe_service.py`. Emails in `ADMIN_EMAILS` are auto-promoted to Pro.
- Jobs: `services/worker.py` runs a ThreadPoolExecutor that calls `EarCopyEngine(mode="ai_inst")` via `services/transcribe.py`. Tier logic: `free` truncates input to 30s; `pro` additionally exports per-track MIDIs (piano/bass/drums by channel). MP3 preview is available on all tiers.
- Persistence: SQLAlchemy + SQLite (`data/noion.db`), models `User`/`Subscription`/`Job`. All config via env vars in `webapp/config.py` (SECRET_KEY, STRIPE_*, DATABASE_URL, MAX_WORKERS, ...).
- `frontend/` — React 18 + TypeScript + Vite + Tailwind, state via zustand (`store.ts`), API client in `api.ts`.

### Quality system
Output quality is governed by numeric targets T1–T11 (duration, peak, clipping, RMS, 6-band frequency balance, onset ratio, BPM, per-part velocity, stereo correlation, LR energy):
- `tests/test_quality_targets.py` — end-to-end post-AI pipeline against `fixtures/test.mp3`, asserts the targets.
- `scripts/measure_quality.py` — CLI to measure any input/output pair against the same targets.
- `docs/quality_baseline.md` — recorded baseline with per-phase improvement tables. **When changing the synthesis/mastering chain, keep the quality tests passing and update this baseline** (the commit history treats "品質ベースライン更新" as part of such changes).
- `benchmark.py` — manual A/B listening report across 5 genre fixtures.
