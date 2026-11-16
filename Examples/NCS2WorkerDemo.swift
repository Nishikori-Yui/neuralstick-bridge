import Foundation
import Darwin

@main
enum NCS2WorkerDemo {
    static func main() {
        do { try run() }
        catch {
            FileHandle.standardError.write(Data("NCS2WorkerClient: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        var worker = root.appendingPathComponent("build-arm64/ncs2_worker")
        var plugins = root.appendingPathComponent("vendor/openvino-2022.3/bin/arm64/Release/plugins.xml")
        var model: URL?
        var timeout = 15.0, startupTimeout = 30.0, exitTimeout = 3.0
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments == ["--help"] {
            print("Usage: NCS2WorkerClient --model XML [--worker PATH] [--plugins PATH] [--timeout SECONDS] [--startup-timeout SECONDS] [--exit-timeout SECONDS]")
            print("Native macOS arm64 only. Runs one MYRIAD inference on a static single-input/single-output f32 model; no CPU fallback.")
            return
        }
        var index = 0
        var seen = Set<String>()
        while index < arguments.count {
            let option = arguments[index]
            guard index + 1 < arguments.count, seen.insert(option).inserted else {
                throw NCS2ClientError("Missing value or duplicate option: \(option)")
            }
            let value = arguments[index + 1]
            switch option {
            case "--worker": worker = URL(fileURLWithPath: value)
            case "--plugins": plugins = URL(fileURLWithPath: value)
            case "--model": model = URL(fileURLWithPath: value)
            case "--timeout", "--startup-timeout", "--exit-timeout":
                guard let seconds = Double(value) else { throw NCS2ClientError("Invalid timeout: \(value)") }
                _ = try PipeDeadline(seconds: seconds)
                if option == "--timeout" { timeout = seconds }
                else if option == "--startup-timeout" { startupTimeout = seconds }
                else { exitTimeout = seconds }
            default: throw NCS2ClientError("Unknown option: \(option)")
            }
            index += 2
        }
        guard let model else { throw NCS2ClientError("--model is required") }
        let client = try NCS2WorkerClient(worker: worker, plugins: plugins, model: model, timeout: timeout,
                                          startupTimeout: startupTimeout, exitTimeout: exitTimeout)
        do {
            guard let ready = client.ready else { throw NCS2ClientError("Worker has no ready descriptor") }
            print("ready: schema_version=1 case=\(ready.case) device=MYRIAD input=f32\(ready.input.shape) output=f32\(ready.output.shape)")
            print("health: ok completed=\(try client.health())")
            let input = (0..<(try ready.input.elementCount())).map { Float(($0 * 37) % 65 - 32) / 16 }
            let result = try client.infer(input)
            print("infer: hardware_executed=true outputs=\(result.output.count) infer_ms=\(result.milliseconds)")
            let status = try client.shutdown()
            print("shutdown: ok worker_exit_code=\(status)")
            print("SWIFT_WORKER_RESULT=PASS")
        } catch {
            do { try client.close() }
            catch let cleanupError { throw NCS2ClientError("\(error); cleanup failed: \(cleanupError)") }
            throw error
        }
    }
}
