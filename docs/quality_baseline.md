# Quality Baseline (Phase 3a)

Measured: 2026-05-04
Input: `fixtures/test.mp3` (15s, 44100Hz, stereo)
Pipeline: synthetic notes -> post-AI pipeline (consolidate -> pad -> humanize -> MIDI -> FluidSynth -> master)
Commit: Phase 2 completed (`ca9b600`)

## Results

```
Item                              Value            Target  Status
------------------------------------------------------------------
T1  Duration (s)                15.2000      [14.7, 15.3]    PASS
T2  Peak                         0.9192           <= 0.95    PASS
T3  Clipped samples                   0              <= 0    PASS
T4  Silence (%)                  0.0000       [5.0, 15.0]    FAIL
T5  RMS                          0.3340      [0.18, 0.32]    FAIL
T6a Sub (<60Hz)                  3.8606        [0.7, 1.5]    FAIL
T6b Bass (60-250Hz)              1.1908        [0.7, 1.4]    PASS
T6c LMid (250-800Hz)             6.5677        [0.7, 1.5]    FAIL
T6d Mid (800-2kHz)           49582.0807        [0.7, 1.4]    FAIL
T6e HMid (2k-8kHz)           104662.3813        [0.6, 1.4]    FAIL
T6f Air (>8kHz)               2875.6838        [0.5, 1.3]    FAIL
T7  Onset ratio                  2.2509           >= 0.85    PASS
T8  BPM diff                   114.4037            <= 8.0    FAIL
T9a Vel melody                  64.6471         [70, 127]    FAIL
T9b Vel chord                   35.8280         [65, 127]    FAIL
T9c Vel bass                    65.0833         [85, 127]    FAIL
T9d Vel pad mean                22.9444         [55, 127]    FAIL
T9e Vel pad std                  2.8958            >= 5.0    FAIL
T10 Stereo corr                 -0.0090           <= 0.95    PASS
------------------------------------------------------------------
Result: 6 PASS, 13 FAIL, 0 N/A
```

## Analysis

### PASS (6/19)
- **T1 Duration**: Output length within spec
- **T2 Peak**: True Peak limiter working (0.92 <= 0.95)
- **T3 Clip**: Zero clipped samples
- **T6b Bass**: Bass band energy ratio 1.19 within [0.7, 1.4]
- **T7 Onset**: Onset strength 2.25x original (above 0.85 threshold)
- **T10 Stereo**: L/R correlation -0.009 (well separated)

### Critical Failures
| Item | Current | Target | Gap |
|------|---------|--------|-----|
| T4 Silence | 0% | 5-15% | No natural pauses in synthesis |
| T5 RMS | 0.334 | 0.18-0.32 | 4% over, master gain too hot |
| T6a Sub | 3.86 | 0.7-1.5 | Sub energy 3.9x input |
| T6c LMid | 6.57 | 0.7-1.5 | LMid energy 6.6x input (piano fundamental region) |
| T6d-f Mid/HMid/Air | 49k-105k | ~1.0 | Input fixture has near-zero energy above 800Hz; ratios are infinity-like |
| T8 BPM diff | 114.4 | <= 8.0 | Input=63.8 BPM vs output=178.2 BPM (synthetic 120BPM tempo, doubled by beat tracker) |
| T9a-e Velocity | 23-65 | 55-85+ | Post-humanize velocity too low across all parts |

### Notes on T6d-f
The test fixture (`fixtures/test.mp3`) contains virtually no energy above 800Hz. This makes the output/input ratio for Mid, HMid, and Air bands meaningless (dividing by near-zero). A fixture with full-spectrum content would produce meaningful ratios. These metrics will become useful when testing with real music input.

### Priority for Phase 3b
1. **T9 Velocity** - Raise velocity levels across all parts (quick fix)
2. **T5 RMS** - Reduce master gain slightly
3. **T4 Silence** - Add natural breathing pauses
4. **T6a/T6c** - Sub/LMid EQ correction
5. **T8 BPM** - Requires real input fixture for meaningful measurement
