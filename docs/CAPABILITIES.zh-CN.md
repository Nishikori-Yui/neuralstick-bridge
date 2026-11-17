# 模型能力与检查

NeuralStick Bridge 支持一份明确的执行契约：在原生 arm64 macOS 主机上，使用 NCS2 的 MYRIAD 插件编译静态 f32 OpenVINO IR。项目不承诺所有 OpenVINO 可读取的计算图都能在 NCS2 上运行。

## 能力矩阵

| 范围 | 支持契约 |
| --- | --- |
| 主机 | 原生 Apple Silicon macOS arm64；不使用 Rosetta 或 x86_64 运行时 |
| 设备 | 通过 `MYRIAD` 访问 Intel Neural Compute Stick 2 / MyriadX |
| 模型文件 | OpenVINO IR XML 及匹配的 BIN，按注册路径读取 |
| 张量类型 | 所有输入和输出均为 `f32` |
| 形状 | 静态，rank 为 1..8 |
| 张量大小 | 单个张量最多 16,384 个元素 |
| schema 1 | 一个输入、一个输出 |
| schema 2 | 1..32 个输入和输出；输入、输出各自聚合上限为 16,384 个元素 |
| Worker 帧 | JSON Lines，每帧最大 1 MiB |
| 设备调度 | 每个 runtime 同时执行一个推理；FIFO 队列最多等待 32 个任务 |
| 取消 | 仅可取消排队任务；运行中的设备推理继续执行 |
| 回退 | 不回退到 CPU |
| 摄像头 | 不包含摄像头集成 |

schema 2 表示一次模型调用包含多个带索引的输入和输出张量，不会把多个任务合并成一个 batch。动态形状、非 f32 张量，以及被锁定版 MYRIAD 编译器拒绝的计算图均不在 worker 契约内。

可通过以下命令获取机器可读矩阵：

```sh
./ncs2 capabilities --json
```

该输出不依赖 USB 状态，可供支持工具在设备配置前显示。

## 离线检查模型

先构建原生模型检查器，再检查 XML：

```sh
./ncs2 build-probe
./ncs2 inspect --model models/classifier.xml
```

检查器使用锁定的 OpenVINO runtime 读取 IR，报告每个端口的索引、名称、数据类型、静态形状状态、rank、shape 和元素数，并分别判断 schema 1 与 schema 2 的限制。检查过程不会获取 NCS2 设备锁、打开 USB、启动固件或针对 MYRIAD 编译计算图。兼容结果表示模型满足 Bridge 的结构限制；首次启动 worker 时仍会请求 MYRIAD 编译计算图。

Python 应用可调用相同报告接口：

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

Swift 应用使用异步 API：

```swift
let inspector = try NCS2ModelInspector(
    executable: root.appendingPathComponent("build-arm64/ncs2_model_inspect")
)
let report = try await inspector.inspect(xml: modelURL)
print(report.schema2.compatible, report.inputs.first?.shape as Any)
```

检查器是独立的原生可执行文件，因为 OpenVINO 模型读取器需要留在兼容运行时边界内。只安装 Python wheel 或 Swift package 的应用还需要单独提供匹配的检查器程序。

构建原生工具后，可显式运行真机 SDK 契约测试：

```sh
NCS2_RUN_HARDWARE_TESTS=1 python3 -B -m unittest discover -s python/tests -p test_hardware_sdk.py -v
```

测试在真实设备上运行五次 schema 2 ReLU 推理，并检查输出、推理计时、worker 复用和结果释放。需要 USB 访问权限。
