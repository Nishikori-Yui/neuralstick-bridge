# Model capabilities and inspection

NeuralStick Bridge supports one specific execution contract: static f32 OpenVINO IR on a native arm64 macOS host, compiled for the NCS2 MYRIAD plugin. It does not promise compatibility with every graph accepted by OpenVINO.

## Capability matrix

| Area | Supported contract |
| --- | --- |
| Host | Native Apple Silicon macOS arm64; no Rosetta or x86_64 runtime |
| Device | Intel Neural Compute Stick 2 / MyriadX through `MYRIAD` |
| Model files | OpenVINO IR XML and matching BIN, read from their registered path |
| Tensor dtype | `f32` for every input and output |
| Shape | Static, rank 1..8 |
| Tensor size | At most 16,384 elements per tensor |
| Schema 1 | Exactly one input and one output |
| Schema 2 | 1..32 inputs and 1..32 outputs; at most 16,384 aggregate input and output elements |
| Worker frame | JSON Lines, maximum 1 MiB per frame |
| Device scheduling | One active inference per runtime; FIFO queue with at most 32 waiting tasks |
| Cancellation | Queued tasks only; running device inference continues |
| Fallback | No CPU fallback |
| Camera | No camera integration |

Schema 2 means one model invocation with multiple indexed input and output tensors. It does not combine separate tasks into a larger batch. Dynamic shapes, non-f32 tensors, and graphs rejected by the pinned MYRIAD compiler are outside the worker contract.

The machine-readable form is available with:

```sh
./ncs2 capabilities --json
```

The output is independent of USB state and is suitable for support tools to display before device setup.

## Inspect a model offline

Build the native model inspector with the rest of the C++ tools, then inspect the XML:

```sh
./ncs2 build-probe
./ncs2 inspect --model models/classifier.xml
```

The inspector reads the IR using the pinned OpenVINO runtime and reports each port's index, name, dtype, static-shape status, rank, shape and element count. It evaluates the schema 1 and schema 2 limits separately. It does not acquire the NCS2 device lock, open USB, boot firmware or compile the graph for MYRIAD. A compatible result means the model meets the Bridge's structural protocol limits; the first worker startup still asks MYRIAD to compile the graph.

Python applications can consume the same report:

```python
from pathlib import Path
from neuralstick_bridge import inspect_model

report = inspect_model(
    Path("models/classifier.xml"),
    executable=Path("build-arm64/ncs2_model_inspect"),
)
print(report.supported_schemas)
print(report.inputs[0].dtype, report.inputs[0].shape)
```

Swift applications use the async API:

```swift
let inspector = try NCS2ModelInspector(
    executable: root.appendingPathComponent("build-arm64/ncs2_model_inspect")
)
let report = try await inspector.inspect(xml: modelURL)
print(report.schema2.compatible, report.inputs.first?.shape as Any)
```

The inspector is a separate native executable because OpenVINO's model reader remains inside the compatibility runtime boundary. Applications that install only the Python wheel or Swift package must supply the matching inspector executable separately.

To run the opt-in physical-device SDK contract test after building the native tools, use:

```sh
NCS2_RUN_HARDWARE_TESTS=1 python3 -B -m unittest discover -s python/tests -p test_hardware_sdk.py -v
```

It runs five schema-2 ReLU tasks and checks output values, inference timers, worker reuse and result release. USB access is required.
