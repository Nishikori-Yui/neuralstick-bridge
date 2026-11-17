# Camera-free performance workloads

The existing `bench` command measures a deterministic 1x4 ReLU graph. It is useful for checking device startup, a basic MYRIAD execution path, and timing stability; its result does not predict application throughput. The SDK now exposes task-level and worker inference timers so workload experiments can distinguish model/worker startup from a warm inference.

## Useful experiments

| Experiment | What it answers | Workload shape |
| --- | --- | --- |
| Cold start versus warm reuse | How much time is spent launching a worker and compiling an IR compared with repeated inference? | Submit one task, then many tasks for the same registered model; compare `worker_start_count`, `last_task_milliseconds`, and `last_inference_milliseconds`. |
| Model-switch cost | Is keeping one model resident worth the compile and shutdown cost when applications alternate models? | Compare long runs grouped by model with strict alternating submissions; count model starts and switches. |
| Host/worker transport overhead | How much end-to-end time does JSON encoding, pipe transport, validation and result decoding add around the device call? | Sweep input/output element counts using real static IR, comparing SDK task time with worker inference time. Treat their difference as combined wrapper/runtime overhead, not pure USB latency. |
| Tensor-size scaling | Where does transfer and serialization cost begin to dominate compute? | Use supported static f32 signal-processing or tabular models with increasing tensor sizes up to the protocol limit. Keep the operation graph and values fixed while changing size. |
| Queue latency and back-pressure | How does an application behave when producers submit faster than one NCS2 can execute? | Submit at controlled rates; record queue depth, queue-full counts, task wait time, completion latency and result-retention use. |
| Sustained thermal and recovery behavior | Does latency drift during a long run, and how quickly does a restarted worker recover after a controlled fault? | Repeat a fixed workload while recording monotonic timestamps, per-task latency, worker restarts and USB reconnect observations. |
| USB topology and power | Does a direct port, powered hub or USB 2/3 path change stability or measured latency? | Repeat the same model and input sequence for each topology; record negotiated link, hub, power source, ambient conditions and device identity without publishing serial numbers. |

## Camera-free application workloads

- **Audio keyword spotting:** feed precomputed MFCC or log-mel windows from audio files. This isolates NCS2 inference from microphone capture and feature extraction.
- **Sensor anomaly scoring:** run a small static model over vibration, temperature or machine telemetry windows. The output can be a score or a compact class vector.
- **Tabular classification:** test fixed feature vectors for local categorization, quality checks or rules that are expensive to maintain by hand.
- **Text classification:** try a compact static classifier on token or embedding vectors. Validate its operators and tensor sizes with `inspect` and an actual MYRIAD compile before benchmarking it.
- **Signal transforms:** evaluate small learned denoisers or regressors over audio/sensor blocks, with host preprocessing timed separately.

These are candidate workloads, not claims that a particular imported model will compile. Use a reproducible IR/BIN pair, fixed inputs, an independent output reference, warm-up iterations, and enough samples to report p50/p95 as well as min/mean/max. Separate first-use compile time from warm inference. Record the exact model hash, worker protocol, OpenVINO revision, host, USB topology and power setup with each result.

## Reading SDK measurements

`last_task_milliseconds` covers worker startup or model switching, model loading/compilation, request transport and validation for that scheduled task. It excludes time spent waiting in the SDK queue. `last_inference_milliseconds` is measured in the native worker from `start_async()` through `wait_for()`; it excludes JSON encoding and output serialization. Schema 1 and schema 2 report this timer.

The difference between those timers groups several host-side costs and should not be labeled USB latency. A controlled transport study needs additional timestamps around input serialization, pipe write/read, worker tensor copies and output serialization. Those measurements belong in a separate lab harness rather than the default application SDK.
