# 异步应用 SDK

SDK 在现有 schema 1/2 同步客户端之上增加进程内模型注册和有界 FIFO 队列。Python 使用 `asyncio`，Swift 通过 `NCS2Kit` 暴露 actor 隔离的 `async/await` API。两端启动同一套 arm64 worker，把锁定的 OpenVINO ABI 留在应用进程之外。

## Python

Python SDK 需要 Python 3.11+，运行时只依赖标准库：

```sh
python3 -m pip install .
# 开发工具，包括 JSON Schema 校验：
python3 -m pip install -e '.[dev]'
```

安装后可用 `from neuralstick_bridge import NCS2Runtime`。同一 wheel 保留 `ncs2_async`、`ncs2_client` 和 `ncs2_multi_client` 导入。安装过程不下载 worker、OpenVINO 或固件，它们需单独构建或提供。


```python
import asyncio
from pathlib import Path
from ncs2_async import NCS2Runtime

async def main():
    async with NCS2Runtime(
        worker=Path("build-arm64/ncs2_worker"),
        multi_worker=Path("build-arm64/ncs2_multi_worker"),
        plugins=Path("vendor/openvino-2022.3/bin/arm64/Release/plugins.xml"),
    ) as runtime:
        await runtime.register_model("classifier", Path("models/classifier.xml"), schema=2)
        handle = await runtime.submit("classifier", [[0.0] * 128])
        try:
            result = await handle.result()
            print(result.outputs[0].shape, result.outputs[0].data)
        finally:
            await handle.release()

asyncio.run(main())
```

输入始终是“每个输入张量对应一个序列”的数组，即使 schema 1 也如此。所有公开操作必须在首次使用该 runtime 的同一个 event loop 中执行。同步的 worker 启动、推理和关闭通过 `asyncio.to_thread` 执行，不阻塞 event loop。

## Swift

`NCS2Kit` SwiftPM product 支持 macOS 13+ arm64，不包含运行时或固件二进制：

```swift
import Foundation
import NCS2Kit

func classify(root: URL, model: URL, values: [Float]) async throws {
    let runtime = try NCS2Runtime(
        worker: root.appendingPathComponent("build-arm64/ncs2_worker"),
        multiWorker: root.appendingPathComponent("build-arm64/ncs2_multi_worker"),
        plugins: root.appendingPathComponent("vendor/openvino-2022.3/bin/arm64/Release/plugins.xml")
    )
    do {
        try await runtime.registerModel("classifier", xml: model, schema: .multi)
        let handle = try await runtime.submit(model: "classifier", inputs: [values])
        let result = try await handle.result()
        print(result.outputs[0].data)
        try await handle.release()
        try await runtime.close()
    } catch {
        try? await runtime.close()
        throw error
    }
}
```

runtime actor 持有调度和结果状态；独立的串行 I/O executor 负责非 `Sendable` 的同步 worker 客户端。阻塞管道操作不会运行在 actor 或主线程上。

## 模型与 worker 复用

- 注册唯一名称、XML 路径和 schema（1 或 2；默认是 2）。
- 注册只保存元数据，不编译模型，也不复制 XML/BIN。使用期间必须保持 XML 与匹配的 BIN 位于注册路径。
- ready 描述符是 shape 校验的权威来源。
- SDK 在接受任务时复制输入。shape 不匹配会在模型加载后、推理前让任务失败。
- 相邻任务使用同一个模型时复用 worker；模型变化时按 FIFO 顺序关闭旧 worker，再启动新 worker。
- 静态 f32、端口数、帧大小和聚合元素数限制沿用 worker。

## 队列和任务生命周期

任务状态包括 `queued`、`running`、`succeeded`、`failed` 和 `cancelled`。同一个 runtime 同时只执行一个任务，提交顺序就是执行顺序。

`runtime.status()` 返回一个时间点快照，包括等待队列长度、运行中的任务和模型、已加载的 worker 模型、设备租约状态、已注册模型数量以及关闭状态。它不会访问 USB，也不会强制启动 worker。

默认容量是 **32 个等待任务**，不包含当前运行任务。队列满时立即返回 `queue_full`，不会保留被拒绝的任务。

`cancel()` 只有在成功移除排队任务时返回 true。运行中或终态任务返回 false。取消 `result()` 的等待操作只解除当前等待者，不会终止硬件推理。

结果和任务错误会一直保存在 runtime 内存中，直到调用 `release()`。终态任务才可以释放；释放后 handle 的访问会返回 `result_released`。及时释放不再需要的结果可以限制内存占用。

`close()` 会拒绝新任务、取消排队任务、等待当前推理、关闭并回收 worker、释放设备租约。调用者取消 close 等待也不会放弃清理。已完成结果在 close 后仍可读取，直到显式释放。

## 设备占用与错误

SDK 在 `/tmp/neuralstick-bridge-device-UID.lock.sdk` 持有调度租约，直到关闭。native worker 和 probe 在访问 MYRIAD 期间持有去掉 `.sdk` 后缀的设备锁。两个锁分开，避免 worker 和自己的父进程竞争。同步客户端由 native worker 保护，外部 OpenVINO 程序不参与此协议。

`NCS2_DEVICE_LOCK`、`device_lock` 或 `deviceLock` 可以指定设备锁路径，SDK 会传给子 worker。模型切换的间隙，竞争 probe 可能取得设备锁；后续启动会返回 `device_busy`，不会自动重试推理。`--validate-only` 不获取锁。不安装 daemon 或数据库。

稳定错误分类包括：

| 错误 | 含义 |
| --- | --- |
| `queue_full` | 队列已满，没有任务入队。 |
| `device_busy` / `device_unavailable` | 设备已被另一个 SDK runtime 占用，或锁无法获取。 |
| `model_load_failed` | 模型、运行时、设备启动或 ready 元数据失败。 |
| `invalid_argument` | 配置、数值或 shape 无效。 |
| `protocol_error` | 坏帧、截断响应、传输超时或生命周期错误。 |
| `inference_failed` | worker 返回推理错误；fatal 错误会回收 worker。 |
| `unknown_model` / `model_exists` | 模型注册名称错误。 |
| `task_cancelled` / `task_not_finished` / `result_released` | 任务句柄生命周期错误。 |
| `closed` | runtime 已关闭。 |

schema 1 才提供 worker 推理计时；schema 2 的计时字段为空。模型编译和队列等待不计入该字段。

## 兼容性

worker 协议保持不变。已有同步 Python/Swift 客户端继续可用。Python 使用 `NCS2FatalError` 区分 fatal 响应，Swift 的 `NCS2ClientError` 提供 `isFatalResponse`，不需要解析厂商错误文本。

状态快照中的 `last_error`（Python）或 `lastError`（Swift）保留最近一次任务失败，排队取消不计入。释放任务结果不会清除该诊断。`device_owned` / `deviceOwned` 表示 SDK 调度租约；硬件锁由 native worker 持有。快照反映进程内状态，不是实时 USB 健康探测。
