# EvolveTS headroom scan results (TASK_015)

## What this is

The TASK_015 headroom scan over the four seeded scan presets of increasing severity (`synthetic-shift1` to `synthetic-shift3`, 100 series each) and the wide replicate of the default grammar (`synthetic-wide`, 360 series), plus the three dev datasets of the first Gate 1 run (ETTh1, ECL, `synthetic-dev`), each tuned on the `simple` and `simple+foundation` pools with the same grids and budgets as the reference run. Headroom is (best tuned baseline MASE − target MASE) / best tuned baseline MASE per pool, against four hindsight targets: the switching oracles with at most 8 and 32 switches, the best expert per series, and (on synthetic pools only) the true-best forecast, each with a paired bootstrap interval over series. The recommendation below is made by the rule fixed in `docs/tasks/TASK_015_headroom_scan.md` before any run; the final choice is the owner's.

## Ranked headroom

| rank | dataset | pool | series | best tuned baseline | baseline MASE | switch-8 headroom | 95% interval | switch-32 headroom | best-per-series headroom | true-best headroom | replay seconds | tune seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | synthetic-shift3 | simple+foundation | 100 | fixed_share | 1.3120 | 0.0754 | [0.0636, 0.0899] | 0.1292 | 0.0050 | 0.3471 | 163.0721 | 12917.3126 |
| 2 | synthetic-shift3 | simple | 100 | fixed_share | 1.4802 | 0.0716 | [0.0593, 0.0859] | 0.1306 | -0.0428 | 0.4213 | 105.9297 | 8602.0144 |
| 3 | synthetic-shift2 | simple | 100 | fixed_share | 1.1563 | 0.0605 | [0.0534, 0.0679] | 0.1019 | -0.0356 | 0.2944 | 103.3386 | 8482.0894 |
| 4 | synthetic-shift2 | simple+foundation | 100 | fixed_share | 1.0493 | 0.0580 | [0.0524, 0.0642] | 0.0969 | 0.0049 | 0.2224 | 163.3748 | 12716.4374 |
| 5 | synthetic-shift1 | simple+foundation | 100 | fixed_share | 0.8675 | 0.0464 | [0.0423, 0.0507] | 0.0773 | 0.0051 | 0.1406 | 155.7409 | 12126.3565 |
| 6 | synthetic-shift1 | simple | 100 | fixed_share | 0.9507 | 0.0501 | [0.0418, 0.0601] | 0.0868 | -0.0214 | 0.2158 | 106.9345 | 8366.0214 |
| 7 | synthetic-dev | simple+foundation | 120 | fixed_share | 0.7771 | 0.0384 | [0.0351, 0.0419] | 0.0697 | 0.0047 | 0.0704 | 154.8817 | 12586.9865 |
| 8 | synthetic-wide | simple+foundation | 360 | fixed_share | 0.7822 | 0.0348 | [0.0330, 0.0365] | 0.0616 | 0.0048 | 0.0703 | 607.7434 | 112952.2528 |
| 9 | synthetic-wide | simple | 360 | fixed_share | 0.8549 | 0.0303 | [0.0280, 0.0327] | 0.0655 | -0.0177 | 0.1494 | 388.6531 | 46933.3160 |
| 10 | synthetic-dev | simple | 120 | fixed_share | 0.8478 | 0.0316 | [0.0280, 0.0356] | 0.0708 | -0.0216 | 0.1480 | 109.2854 | 8291.3796 |
| 11 | etth1 | simple | 7 | fixed_share | 0.9251 | 0.0061 | [-0.0102, 0.0323] | 0.0647 | -0.0694 | n/a | 37.4052 | 2907.4636 |
| 12 | ecl | simple | 20 | fixed_share | 0.9201 | -0.0193 | [-0.0313, -0.0047] | 0.0209 | -0.0613 | n/a | 148.1843 | 11761.9911 |

## Ranked headroom, leakage-risk pools

| rank | dataset | pool | series | best tuned baseline | baseline MASE | switch-8 headroom | 95% interval | switch-32 headroom | best-per-series headroom | true-best headroom | replay seconds | tune seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | etth1 | simple+foundation | 7 | hedge | 0.8133 | 0.0270 | [0.0228, 0.0356] | 0.0729 | -0.0026 | n/a | 52.6897 | 4290.9561 |
| 2 | ecl | simple+foundation | 20 | hedge | 0.6837 | 0.0164 | [0.0118, 0.0236] | 0.0400 | -0.0051 | n/a | 216.2954 | 17325.4187 |

## Recommendation

By the rule fixed in TASK_015, the recommended Phase 1 setting is synthetic-shift3, pool simple+foundation (switch-8 headroom 0.0754, 95 % interval [0.0636, 0.0899]; one evaluation takes 163.0721 seconds).

Also qualifying: synthetic-shift2, pool simple+foundation (switch-8 headroom 0.0580, 95 % interval [0.0524, 0.0642]; one evaluation takes 163.3748 seconds).

Also qualifying: synthetic-shift1, pool simple+foundation (switch-8 headroom 0.0464, 95 % interval [0.0423, 0.0507]; one evaluation takes 155.7409 seconds).

Also qualifying: synthetic-dev, pool simple+foundation (switch-8 headroom 0.0384, 95 % interval [0.0351, 0.0419]; one evaluation takes 154.8817 seconds).

Also qualifying: synthetic-wide, pool simple+foundation (switch-8 headroom 0.0348, 95 % interval [0.0330, 0.0365]; one evaluation takes 607.7434 seconds).

The final choice of the Phase 1 setting is the owner's.

## Caveats

- Dev numbers: the baselines were tuned, and the targets computed, on these same series.
- The switching oracles and the best expert per series are single-expert hindsight bounds: no online updater can reach them, and they do not bound weighted combinations.
- ETTh1 and ECL results with foundation models are leakage-risk (a foundation model may have seen these datasets in pretraining); they are ranked in a separate table.
- Several comparisons, uncorrected: there is no multiple-comparison correction.
- The baselines' grid-edge caveat from the first run stands: several selected configurations sit at the edge of the tuning grid (Hedge eta 0.015625 on ETTh1 simple and pooled, 0.03125 on ECL pooled), so those baselines may be slightly under-tuned (`docs/results/gate1-dev-first-run.md`).
- The scan presets were designed in TASK_015 before any run, and none was changed after.

## Follow-up decisions for the owner

- Hard real domains (#125 candidate 2; the open questions are listed in `docs/tasks/TASK_015_headroom_scan.md`).
- Larger or more varied expert pools (a `CACHE_VERSION` bump).
- Updaters that output forecasts.
- Probabilistic metrics (CRPS, calibration).

## Provenance (for reproduction)

**Tune runs in the report** (all `completed`, condition `gate1-dev-tune`, `validate_run` none, per the report's Runs table):

| run id | datasets | records | recorded git commit | commit mapping |
| --- | --- | --- | --- | --- |
| gate1-tune-20261013T151300Z-b36261f7 | etth1, ecl, synthetic-dev (reference run) | 488 | 575906467740659c72c91aa7ca8962cc9d771b6c | tarball of main `7a56750ff68e8aee9a6a3c040695a842839c2507` (see `docs/results/gate1-dev-first-run.md`) |
| gate1-scan-synthetic-shift1-20261017T014154Z-41ad1404 | synthetic-shift1 | 164 | 95ba71178c31cace75d48448876c5ae6d9f5b7a0 | tarball of main `41ad14040997a041861b3fc9ff0fc4483e419790` (#137; the run id's `41ad1404` and the W9 `::warning` line both record it) |
| gate1-scan-synthetic-shift2-d1 | synthetic-shift2 | 164 | 6a5820350620fcc0ab139886a58be453522e80cd | real main commit (#138; GitHub lookup HTTP 200) |
| gate1-scan-synthetic-shift3-d1 | synthetic-shift3 | 164 | 6a5820350620fcc0ab139886a58be453522e80cd | real main commit (#138; GitHub lookup HTTP 200) |
| gate1-scan-synthetic-wide-d1 | synthetic-wide | 164 | 6a5820350620fcc0ab139886a58be453522e80cd | real main commit (#138; GitHub lookup HTTP 200) |

**W8 foundation builds** (workflow `gpu-foundation-cache.yml`; each uploads its manifest as the artifact `foundation-manifest-<dataset>-<model>`, hashes only, 30-day retention):

| dataset | model | per_family | workflow run | outcome |
| --- | --- | --- | --- | --- |
| synthetic-shift1 | chronos2 | 20 | 38743569519 | success |
| synthetic-shift1 | tirex | 20 | 38763536102, 38789228132, then 38861035436 | first two failed (OOM kill on sp-1; retries added by #137), last succeeded |
| synthetic-shift1 | timesfm25 | 20 | 38772242571, then 38805662460 | first failed, retry succeeded |
| synthetic-shift2 | chronos2 | 20 | 38772225956 | success |
| synthetic-shift2 | tirex | 20 | 38743573110 | success |
| synthetic-shift2 | timesfm25 | 20 | 38772258311 | success |
| synthetic-shift3 | chronos2 | 20 | 38772250476 | success |
| synthetic-shift3 | tirex | 20 | 38772234522 | success |
| synthetic-shift3 | timesfm25 | 20 | 38743577239 | success |
| synthetic-wide | chronos2 | 60 | 38763532172 | success |
| synthetic-wide | tirex | 60 | 38772266388 | success |
| synthetic-wide | timesfm25 | 60 | 38772275081 | success |

Run 38860597373 was cancelled before doing any work (no log survives; noted on #135). The scan-dataset choices and the `synthetic-*` build path were added to the workflow by #131; #137 added the retries.

**W9 tune attempts** (workflow `gate1-headroom-tune.yml`):

- 38860604024 (2026-10-16, default datasets, all four pieces in parallel): failed, 'self-hosted runner lost communication' after about 39 min.
- 38960653431 (2026-10-16, default datasets, all four pieces in parallel): failed the same way after about 3.5 h.
- 38982950481 (2026-10-16, `datasets=synthetic-shift1`): success; `TUNE_RUN_IDS=gate1-scan-synthetic-shift1-20261017T014154Z-41ad1404`.
- 38999449134 (2026-10-17): cancelled before doing any work (noted on #135).
- 39018424897 (2026-10-17, `datasets=synthetic-shift2`): failed, 'self-hosted runner lost communication'.
- 39044109734 (2026-10-17, `datasets=synthetic-shift2`, re-dispatch): failed the same way.

The failed attempts started pieces that were left incomplete in the lab node's ledger and are not used by the report; their tune run ids are not recoverable from the surviving logs, because W9 prints `TUNE_RUN_IDS` only after `wait`, which never ran (the attempts and their datasets are recorded above and on #135). After the fourth loss, `synthetic-shift2`, `synthetic-shift3` and `synthetic-wide` were run detached from the runner with `.github/scripts/headroom-detached.sh` (#138), tag `d1`, started on sp-1 via `docker exec -d` because runner-started processes get `oom_score_adj` 500 and sp-1 was under memory pressure from another user's job.

**Commands and seeds.**

- W9 piece (per dataset `d`, updater seed 0): `evolvets gate1 tune --dataset "$d" --pool-spec simple --pool-spec simple+foundation --run-id <id> --max-wall-seconds 172800 --seed 0 --ledger-root /data/evolve/ledgers --cache-root /data/evolve/cache_store --foundation-root /data/evolve/foundation_cache --data-root /data/evolve/data --manifest-dir data/manifests`.
- Detached pieces: the same command, run through `.github/scripts/headroom-detached.sh d1 <dataset>` (seed 0, `--max-wall-seconds 172800`).
- W10 report (bootstrap seed 0): `evolvets gate1 headroom --ledger-root /data/evolve/ledgers --run gate1-tune-20261013T151300Z-b36261f7 --run gate1-scan-synthetic-shift1-20261017T014154Z-41ad1404 --run gate1-scan-synthetic-shift2-d1 --run gate1-scan-synthetic-shift3-d1 --run gate1-scan-synthetic-wide-d1 --bootstrap-seed 0 --out out/headroom-scan.md`.

**Run URLs.**

- W9 (synthetic-shift1): https://github.com/kostagjorgjievski/evolve/actions/runs/38982950481
- W10 (final report): https://github.com/kostagjorgjievski/evolve/actions/runs/39195980003
- Interim W10 on the partial scan (shift1 plus the reference run only, superseded by the final report): https://github.com/kostagjorgjievski/evolve/actions/runs/39108892008

**Artifact SHA-256.** The `headroom-scan` artifact of W10 run 39195980003:

```
812368c5e21074d14073e2f67c536a7925cb8ef50b3e2cc0442970d05d10fa9b  headroom-scan.md
```

(the digest printed by the run's "Build the report" step; verified with `sha256sum` against the artifact download embedded below).

## The full report

The report below is the W10 artifact, byte-identical.

---

# Headroom scan (dev)

generated by `evolvets gate1 headroom --run gate1-tune-20261013T151300Z-b36261f7 --run gate1-scan-synthetic-shift1-20261017T014154Z-41ad1404 --run gate1-scan-synthetic-shift2-d1 --run gate1-scan-synthetic-shift3-d1 --run gate1-scan-synthetic-wide-d1 --bootstrap-seed 0`

## Runs

| run id | condition | status | git commit | records | validate_run |
| --- | --- | --- | --- | --- | --- |
| gate1-tune-20261013T151300Z-b36261f7 | gate1-dev-tune | completed | 575906467740659c72c91aa7ca8962cc9d771b6c | 488 | none |
| gate1-scan-synthetic-shift1-20261017T014154Z-41ad1404 | gate1-dev-tune | completed | 95ba71178c31cace75d48448876c5ae6d9f5b7a0 | 164 | none |
| gate1-scan-synthetic-shift2-d1 | gate1-dev-tune | completed | 6a5820350620fcc0ab139886a58be453522e80cd | 164 | none |
| gate1-scan-synthetic-shift3-d1 | gate1-dev-tune | completed | 6a5820350620fcc0ab139886a58be453522e80cd | 164 | none |
| gate1-scan-synthetic-wide-d1 | gate1-dev-tune | completed | 6a5820350620fcc0ab139886a58be453522e80cd | 164 | none |

## Protocol versions

| protocol | version |
| --- | --- |
| ledger | 1.0 |
| arena | 3 |
| generator | 1 |
| cache_manifest | 1 |

## Ranked headroom

| rank | dataset | pool | series | best tuned baseline | baseline MASE | switch-8 headroom | 95% interval | switch-32 headroom | best-per-series headroom | true-best headroom | replay seconds | tune seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | synthetic-shift3 | simple+foundation | 100 | fixed_share | 1.3120 | 0.0754 | [0.0636, 0.0899] | 0.1292 | 0.0050 | 0.3471 | 163.0721 | 12917.3126 |
| 2 | synthetic-shift3 | simple | 100 | fixed_share | 1.4802 | 0.0716 | [0.0593, 0.0859] | 0.1306 | -0.0428 | 0.4213 | 105.9297 | 8602.0144 |
| 3 | synthetic-shift2 | simple | 100 | fixed_share | 1.1563 | 0.0605 | [0.0534, 0.0679] | 0.1019 | -0.0356 | 0.2944 | 103.3386 | 8482.0894 |
| 4 | synthetic-shift2 | simple+foundation | 100 | fixed_share | 1.0493 | 0.0580 | [0.0524, 0.0642] | 0.0969 | 0.0049 | 0.2224 | 163.3748 | 12716.4374 |
| 5 | synthetic-shift1 | simple+foundation | 100 | fixed_share | 0.8675 | 0.0464 | [0.0423, 0.0507] | 0.0773 | 0.0051 | 0.1406 | 155.7409 | 12126.3565 |
| 6 | synthetic-shift1 | simple | 100 | fixed_share | 0.9507 | 0.0501 | [0.0418, 0.0601] | 0.0868 | -0.0214 | 0.2158 | 106.9345 | 8366.0214 |
| 7 | synthetic-dev | simple+foundation | 120 | fixed_share | 0.7771 | 0.0384 | [0.0351, 0.0419] | 0.0697 | 0.0047 | 0.0704 | 154.8817 | 12586.9865 |
| 8 | synthetic-wide | simple+foundation | 360 | fixed_share | 0.7822 | 0.0348 | [0.0330, 0.0365] | 0.0616 | 0.0048 | 0.0703 | 607.7434 | 112952.2528 |
| 9 | synthetic-wide | simple | 360 | fixed_share | 0.8549 | 0.0303 | [0.0280, 0.0327] | 0.0655 | -0.0177 | 0.1494 | 388.6531 | 46933.3160 |
| 10 | synthetic-dev | simple | 120 | fixed_share | 0.8478 | 0.0316 | [0.0280, 0.0356] | 0.0708 | -0.0216 | 0.1480 | 109.2854 | 8291.3796 |
| 11 | etth1 | simple | 7 | fixed_share | 0.9251 | 0.0061 | [-0.0102, 0.0323] | 0.0647 | -0.0694 | n/a | 37.4052 | 2907.4636 |
| 12 | ecl | simple | 20 | fixed_share | 0.9201 | -0.0193 | [-0.0313, -0.0047] | 0.0209 | -0.0613 | n/a | 148.1843 | 11761.9911 |

## Ranked headroom, leakage-risk pools

| rank | dataset | pool | series | best tuned baseline | baseline MASE | switch-8 headroom | 95% interval | switch-32 headroom | best-per-series headroom | true-best headroom | replay seconds | tune seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | etth1 | simple+foundation | 7 | hedge | 0.8133 | 0.0270 | [0.0228, 0.0356] | 0.0729 | -0.0026 | n/a | 52.6897 | 4290.9561 |
| 2 | ecl | simple+foundation | 20 | hedge | 0.6837 | 0.0164 | [0.0118, 0.0236] | 0.0400 | -0.0051 | n/a | 216.2954 | 17325.4187 |

## Per-pool detail

### synthetic-shift3, pool simple+foundation

Best tuned baseline: fixed_share (dev MASE 1.3120), run gate1-scan-synthetic-shift3-d1

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 1.2131 | 0.0989 | 0.0754 | [0.0636, 0.0899] |
| switch_mase_m32 | 1.1424 | 0.1695 | 0.1292 | [0.1096, 0.1529] |
| best expert per series | 1.3055 | 0.0065 | 0.0050 | [0.0020, 0.0079] |
| true-best forecast | 0.8566 | 0.4554 | 0.3471 | [0.2752, 0.4381] |

### synthetic-shift3, pool simple

Best tuned baseline: fixed_share (dev MASE 1.4802), run gate1-scan-synthetic-shift3-d1

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 1.3743 | 0.1060 | 0.0716 | [0.0593, 0.0859] |
| switch_mase_m32 | 1.2869 | 0.1933 | 0.1306 | [0.1100, 0.1542] |
| best expert per series | 1.5436 | -0.0633 | -0.0428 | [-0.0621, -0.0216] |
| true-best forecast | 0.8566 | 0.6236 | 0.4213 | [0.3482, 0.5135] |

### synthetic-shift2, pool simple

Best tuned baseline: fixed_share (dev MASE 1.1563), run gate1-scan-synthetic-shift2-d1

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 1.0863 | 0.0700 | 0.0605 | [0.0534, 0.0679] |
| switch_mase_m32 | 1.0384 | 0.1178 | 0.1019 | [0.0933, 0.1106] |
| best expert per series | 1.1974 | -0.0411 | -0.0356 | [-0.0469, -0.0246] |
| true-best forecast | 0.8159 | 0.3404 | 0.2944 | [0.2634, 0.3268] |

### synthetic-shift2, pool simple+foundation

Best tuned baseline: fixed_share (dev MASE 1.0493), run gate1-scan-synthetic-shift2-d1

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.9884 | 0.0609 | 0.0580 | [0.0524, 0.0642] |
| switch_mase_m32 | 0.9476 | 0.1016 | 0.0969 | [0.0880, 0.1053] |
| best expert per series | 1.0442 | 0.0051 | 0.0049 | [0.0029, 0.0069] |
| true-best forecast | 0.8159 | 0.2334 | 0.2224 | [0.1906, 0.2549] |

### synthetic-shift1, pool simple+foundation

Best tuned baseline: fixed_share (dev MASE 0.8675), run gate1-scan-synthetic-shift1-20261017T014154Z-41ad1404

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.8272 | 0.0403 | 0.0464 | [0.0423, 0.0507] |
| switch_mase_m32 | 0.8004 | 0.0670 | 0.0773 | [0.0708, 0.0837] |
| best expert per series | 0.8631 | 0.0044 | 0.0051 | [0.0041, 0.0062] |
| true-best forecast | 0.7455 | 0.1220 | 0.1406 | [0.1090, 0.1805] |

### synthetic-shift1, pool simple

Best tuned baseline: fixed_share (dev MASE 0.9507), run gate1-scan-synthetic-shift1-20261017T014154Z-41ad1404

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.9031 | 0.0476 | 0.0501 | [0.0418, 0.0601] |
| switch_mase_m32 | 0.8682 | 0.0825 | 0.0868 | [0.0770, 0.0974] |
| best expert per series | 0.9711 | -0.0204 | -0.0214 | [-0.0299, -0.0131] |
| true-best forecast | 0.7455 | 0.2052 | 0.2158 | [0.1821, 0.2599] |

### synthetic-dev, pool simple+foundation

Best tuned baseline: fixed_share (dev MASE 0.7771), run gate1-tune-20261013T151300Z-b36261f7

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.7473 | 0.0298 | 0.0384 | [0.0351, 0.0419] |
| switch_mase_m32 | 0.7229 | 0.0542 | 0.0697 | [0.0640, 0.0758] |
| best expert per series | 0.7734 | 0.0037 | 0.0047 | [0.0040, 0.0055] |
| true-best forecast | 0.7223 | 0.0547 | 0.0704 | [0.0622, 0.0793] |

### synthetic-wide, pool simple+foundation

Best tuned baseline: fixed_share (dev MASE 0.7822), run gate1-scan-synthetic-wide-d1

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.7550 | 0.0272 | 0.0348 | [0.0330, 0.0365] |
| switch_mase_m32 | 0.7340 | 0.0482 | 0.0616 | [0.0583, 0.0649] |
| best expert per series | 0.7785 | 0.0038 | 0.0048 | [0.0043, 0.0053] |
| true-best forecast | 0.7272 | 0.0550 | 0.0703 | [0.0651, 0.0753] |

### synthetic-wide, pool simple

Best tuned baseline: fixed_share (dev MASE 0.8549), run gate1-scan-synthetic-wide-d1

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.8290 | 0.0259 | 0.0303 | [0.0280, 0.0327] |
| switch_mase_m32 | 0.7989 | 0.0560 | 0.0655 | [0.0624, 0.0686] |
| best expert per series | 0.8700 | -0.0151 | -0.0177 | [-0.0204, -0.0152] |
| true-best forecast | 0.7272 | 0.1277 | 0.1494 | [0.1433, 0.1553] |

### synthetic-dev, pool simple

Best tuned baseline: fixed_share (dev MASE 0.8478), run gate1-tune-20261013T151300Z-b36261f7

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.8210 | 0.0268 | 0.0316 | [0.0280, 0.0356] |
| switch_mase_m32 | 0.7878 | 0.0600 | 0.0708 | [0.0659, 0.0759] |
| best expert per series | 0.8661 | -0.0183 | -0.0216 | [-0.0256, -0.0175] |
| true-best forecast | 0.7223 | 0.1255 | 0.1480 | [0.1395, 0.1571] |

### etth1, pool simple

Best tuned baseline: fixed_share (dev MASE 0.9251), run gate1-tune-20261013T151300Z-b36261f7

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.9195 | 0.0056 | 0.0061 | [-0.0102, 0.0323] |
| switch_mase_m32 | 0.8653 | 0.0598 | 0.0647 | [0.0506, 0.0888] |
| best expert per series | 0.9893 | -0.0642 | -0.0694 | [-0.0858, -0.0566] |

### ecl, pool simple

Best tuned baseline: fixed_share (dev MASE 0.9201), run gate1-tune-20261013T151300Z-b36261f7

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.9379 | -0.0178 | -0.0193 | [-0.0313, -0.0047] |
| switch_mase_m32 | 0.9009 | 0.0193 | 0.0209 | [0.0086, 0.0351] |
| best expert per series | 0.9765 | -0.0564 | -0.0613 | [-0.0722, -0.0482] |

### etth1, pool simple+foundation (leakage risk)

Best tuned baseline: hedge (dev MASE 0.8133), run gate1-tune-20261013T151300Z-b36261f7

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.7913 | 0.0219 | 0.0270 | [0.0228, 0.0356] |
| switch_mase_m32 | 0.7540 | 0.0593 | 0.0729 | [0.0632, 0.0862] |
| best expert per series | 0.8154 | -0.0021 | -0.0026 | [-0.0036, -0.0015] |

### ecl, pool simple+foundation (leakage risk)

Best tuned baseline: hedge (dev MASE 0.6837), run gate1-tune-20261013T151300Z-b36261f7

| target | target MASE | gap | headroom | 95% interval |
| --- | --- | --- | --- | --- |
| switch_mase_m8 | 0.6724 | 0.0112 | 0.0164 | [0.0118, 0.0236] |
| switch_mase_m32 | 0.6563 | 0.0273 | 0.0400 | [0.0333, 0.0494] |
| best expert per series | 0.6872 | -0.0035 | -0.0051 | [-0.0078, -0.0020] |

## Notes

- Headroom is (best tuned baseline MASE − target MASE) / best tuned baseline MASE over the pool's series; positive means the target is better. Intervals are paired bootstrap intervals over series (2000 resamples) of the same ratio.
- The baselines were tuned, and the targets computed, on these same dev series.
- The switching oracles and the best expert per series select single experts in hindsight: no online updater can reach them, and they do not bound weighted combinations.
- Replay seconds is the wall-clock of one evaluation (one replay of the selected configuration over every series of the pool); tune seconds is the whole grid.
- Pools are ranked by the lower end of the switch-8 headroom interval; there is no multiple-comparison correction.
