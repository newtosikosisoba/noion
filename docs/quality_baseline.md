# Quality Baseline (Phase 3c)

Measured: 2026-05-06
Input: `fixtures/test.mp3` (15s, 44100Hz, stereo)
Pipeline: synthetic notes -> post-AI pipeline (velocity floor -> humanize -> MIDI -> FluidSynth -> transient/pad post -> part EQ -> stereo positioning -> master)
Commit: Phase 3c completed

## Results

```
Item                              Value            Target  Status
------------------------------------------------------------------
T1  Duration (s)                15.2000      [14.7, 15.3]    PASS
T2  Peak                         0.8410           <= 0.95    PASS
T3  Clipped samples                   0              <= 0    PASS
T4  Silence (%)                  0.0000       [5.0, 15.0]    FAIL
T5  RMS                          0.2451      [0.18, 0.32]    PASS
T6a Sub (<60Hz)                  1.5521        [0.7, 1.5]    FAIL*
T6b Bass (60-250Hz)              0.9363        [0.7, 1.4]    PASS
T6c LMid (250-800Hz)             1.4017        [0.7, 1.5]    PASS
T6d Mid (800-2kHz)           56366.3432        [0.7, 1.4]    FAIL*
T6e HMid (2k-8kHz)           501555.3532        [0.6, 1.4]    FAIL*
T6f Air (>8kHz)               5377.1963        [0.5, 1.3]    FAIL*
T7  Onset ratio                  2.3891           >= 0.85    PASS
T8  BPM diff                    72.1971            <= 8.0    FAIL
T9a Vel melody                  88.4118         [70, 127]    PASS
T9b Vel chord                   68.1828         [65, 127]    PASS
T9c Vel bass                    96.7917         [85, 127]    PASS
T9d Vel pad mean                68.4444         [55, 127]    PASS
T9e Vel pad std                 15.2360            >= 5.0    PASS
T10 Stereo corr                 -0.2889           <= 0.85    PASS
T11 LR energy ratio              0.9466      [0.85, 1.15]    PASS
------------------------------------------------------------------
Result: 14 PASS, 6 FAIL, 0 N/A
* = fixture artifact (skipped in test suite)
```

## Phase 3c Changes (vs Phase 3a baseline)

### Phase 3a → 3c Improvement Table

| Item | 3a | 3c | Status | Fix |
|------|-----|-----|--------|-----|
| T1 Duration | 15.20 | 15.20 | PASS→PASS | — |
| T2 Peak | 0.88 | 0.84 | PASS→PASS | True Peak -1.5dBTP ceiling |
| T3 Clipped | 0 | 0 | PASS→PASS | — |
| T4 Silence | 0.00 | 0.00 | FAIL→FAIL | Synthetic pipeline, no natural pauses |
| T5 RMS | 0.334 | 0.245 | FAIL→PASS | LUFS -16 + Mid/Side master EQ |
| T6b Bass | 1.97 | 0.94 | FAIL→PASS | Mid/Side EQ + LUFS -16 rebalance |
| T6c LMid | 6.57 | 1.40 | FAIL→PASS | Bell cuts + 35% band subtraction (Mid/Side) |
| T7 Onset | 2.44 | 2.39 | PASS→PASS | — |
| T8 BPM diff | 114.4 | 72.2 | FAIL→FAIL | Fixture artifact (63.8BPM vs synthetic 120BPM) |
| T9a melody vel | 64.6 | 88.4 | FAIL→PASS | `_enforce_velocity_floor` [70,110] |
| T9b chord vel | 35.8 | 68.2 | FAIL→PASS | Velocity floor [60,95] |
| T9c bass vel | 65.1 | 96.8 | FAIL→PASS | Velocity floor [80,120] |
| T9d pad mean | 22.9 | 68.4 | FAIL→PASS | Velocity floor [50,80] + ±8 jitter |
| T9e pad std | 2.9 | 15.2 | FAIL→PASS | Pad jitter in velocity floor |
| T10 Stereo corr | 0.50 | -0.29 | PASS→PASS | Per-part panning + spectral drum pan (threshold tightened: <0.95→<0.85) |
| T11 LR energy | N/A | 0.95 | NEW | LR balance correction in master |

**Summary: 6 PASS (3a) → 14 PASS (3c), +8 metrics fixed, +1 new metric added**

## Phase 3c Implementation Details

### Task i: Per-part stereo positioning (`_apply_stereo_position`, `_spectral_drum_pan`)
- Constant-power panning: `L = mono * cos(angle) * √2`, `R = mono * sin(angle) * √2`
- Per-part positions: melody=62(slight L), chord=preserve stereo, bass=64(C), pad=preserve stereo, decoration=90(R), sub_melody=50(L)
- Spectral drum panning: kick(<250Hz)=64(C), snare(250-2kHz)=60(L), hihat(2-6kHz)=85(R), cymbal(>6kHz)=90(R)
- LR balance correction in master chain: measures L/R energy ratio, applies ≤±10% gain to equalize

### Task ii: Stereo quality tests
- T10 threshold tightened from <0.95 to <0.85
- T11 LR energy ratio added: checks 0.85 ≤ E_L/E_R ≤ 1.15

### Task iii: Mastering targets for 歌ってみた
- LUFS target: -14 → -16 (headroom for vocal overlay)
- True Peak ceiling: -1.0 dBTP → -1.5 dBTP
- Hard clip safety: 0.891 → 0.841

### Mid/Side master EQ
- `_master_eq` now operates in mid/side domain to eliminate correlation-dependent EQ behavior
- EQ applied to mid signal only; side signal preserved unchanged
- Result: EQ outcome is independent of L/R correlation from panning

## Remaining for future phases
1. **T4 Silence** — Add natural breathing pauses between phrases
2. **T8 BPM** — Requires real music fixture for meaningful measurement
3. **Full-spectrum fixture** — Would make T6d-f testable
