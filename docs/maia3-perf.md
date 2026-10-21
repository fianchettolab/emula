# Maia-3 performance

How fast and how heavy the bundled Maia-3 model is on real hardware, measured in
[Fianchetto Lab](https://github.com/fianchettolab/fianchetto) ("the app" below).
**Target:** warm inference for the bundled model under 100 ms on a recent iPhone. The
brief set it for 5M, but it now applies to 23M, the model the app ships.

## How to measure

The results below come from a temporary in-app benchmark, `MaiaBenchmarkLab.swift`
(Settings → Lab), removed once they were recorded. To measure again, for example on
another device or after changing the model, restore it from the app's git history (it's in the
commit that removed it) and add its Settings entry back. Run a **Release** build on the
device: Debug builds of Emula are slower, and the results must reflect what users get.

For each Core ML compute-unit setting (`all`, `cpuAndNeuralEngine`, `cpuAndGPU`,
`cpuOnly`), the benchmark:

1. loads the bundled 23M model (`maia3-23m-fp32.mlmodelc`), including one warm-up
   prediction;
2. times 100 raw inferences over four positions (start, opening, middlegame, endgame) and
   reports p50, p95 and max;
3. times 100 full `bestMove` calls, which add game replay, encoding, legal-move masking
   and sampling;
4. reports the memory iOS counts against the app (`phys_footprint`) after loading, and
   the process peak at the end.

The **cold load** is the app's own first load: the Game tab loads the model when the app
starts. To measure it, force-quit the app, wait a moment, relaunch, and run the
benchmark. The very first launch after installing is slower, because Core ML specializes
the model for the device.

Let the device cool between runs; the benchmark prints the thermal state.

## Results

### Devices

| Device | iOS | Cold load | Compute units | Load + warm-up | Inference p50 / p95 | bestMove p50 / p95 | Memory after load | Peak |
|---|---|---|---|---|---|---|---|---|
| iPhone 17 (iPhone18,3) | 27.0.1 | 674 ms | all | 86 ms | 4.9 / 6.5 ms | 4.9 / 5.1 ms | 77 MB | 95 MB |
| | | | cpuAndNeuralEngine | 497 ms | 4.4 / 4.6 ms | 4.6 / 4.8 ms | 83 MB | |
| | | | cpuAndGPU | 395 ms | 5.2 / 6.6 ms | 4.9 / 5.0 ms | 94 MB | |
| | | | cpuOnly | 273 ms | 4.7 / 5.0 ms | 4.6 / 4.8 ms | 90 MB | |

Release build, thermal state nominal, 51 MB in use before the benchmark's own loads.
Max inference stayed under 10 ms for every setting. The first launch after installing
took about 3.0 s to load (measured on a Debug build; the cost is Core ML's
specialization, not our code).

Only an iPhone 17 is available for now. An older device is still to be found.

### Mac, for reference

Apple M3, coremltools 9.0 calling Core ML from Python, fp32 models, batch 1, warm, 50 runs
on an all-zero input. These come from the step 2 export work, not from the benchmark:

| Model | CPU only, p50 / p95 | All compute units, p50 / p95 | Load |
|---|---|---|---|
| 5M | 2.4 / 2.5 ms | 7.7 / 9.6 ms | 0.6 s |
| 23M | 6.7 / 7.0 ms | 7.1 / 12.1 ms | 0.8 s |
| 79M | 20.8 / 21.3 ms | 17.4 / 23.9 ms | 1.2 s |

Simulator runs of the benchmark are only a check that it works: the simulator runs Core ML on
the Mac's CPU, so its numbers don't represent any iPhone.

## Conclusions

- **23M meets the target by a wide margin.** A full move (`bestMove`) takes about 5 ms
  on an iPhone 17 against the 100 ms target, so speed doesn't argue for a smaller model.
  It also leaves headroom for older iPhones, still to be measured.
- **Compute units: keep `all`** (Emula's default, with the `cpuOnly` fallback if
  loading fails). The four settings are within about 1 ms of each other per move, and
  `all` loads fastest once the device has specialized the model (86 ms, against
  273–497 ms for the others).
- **No loading state needed.** The model starts loading at launch, and the cold load
  (about 0.7 s, or about 3 s on the first launch after installing) mostly finishes behind
  the logo animation, which runs for over 3 s. If the first move comes before it's done,
  Maia's turn simply waits for it while the status shows "Maia is thinking…".
- **Memory is modest:** about 77 MB after loading, 95 MB at peak.
