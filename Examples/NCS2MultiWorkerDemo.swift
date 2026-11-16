import Foundation
import Darwin

@main
enum NCS2MultiWorkerDemo {
    static func main() {
        do { try run() }
        catch {
            FileHandle.standardError.write(Data("NCS2MultiWorkerClient: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        var worker = root.appendingPathComponent("build-arm64/ncs2_multi_worker")
        var plugins = root.appendingPathComponent("vendor/openvino-2022.3/bin/arm64/Release/plugins.xml")
        var model: URL?
        var timeout = 15.0, startup = 30.0, exitTimeout = 3.0
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments == ["--help"] {
            print("Usage: NCS2MultiWorkerClient --model XML [--worker PATH] [--plugins PATH] [--timeout SECONDS] [--startup-timeout SECONDS] [--exit-timeout SECONDS]")
            print("Native macOS arm64. Runs one MYRIAD inference on a static multi-tensor f32 model; no CPU fallback.")
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
                guard let seconds = Double(value) else { throw NCS2ClientError("Invalid timeout") }
                _ = try PipeDeadline(seconds: seconds)
                if option == "--timeout" { timeout = seconds }
                else if option == "--startup-timeout" { startup = seconds }
                else { exitTimeout = seconds }
            default: throw NCS2ClientError("Unknown option: \(option)")
            }
            index += 2
        }
        guard let model else { throw NCS2ClientError("--model is required") }
        let client = try NCS2MultiWorkerClient(worker: worker, plugins: plugins, model: model,
                                             timeout: timeout, startupTimeout: startup, exitTimeout: exitTimeout)
        defer { try? client.close() }
        guard let ready = client.ready else { throw NCS2ClientError("Worker has no ready descriptor") }
        let initialHealth = try client.health()
        FileHandle.standardError.write(Data("health: ok completed=\(initialHealth)\n".utf8))
        let inputs = try ready.inputs.map { descriptor in
            (0..<(try descriptor.elementCount())).map { Float(($0 * 37) % 65 - 32) / 16 }
        }
        let outputs = try client.infer(inputs)
        let checked = outputs.reduce(0) { $0 + $1.data.count }
        guard try client.health() == 1 else { throw NCS2ClientError("Completion counter mismatch") }
        let status = try client.shutdown()
        let report: [String: Any] = [
            "SWIFT_MULTI_WORKER_RESULT": "PASS", "schema_version": 2, "device": "MYRIAD",
            "case": ready.case, "input_shapes": ready.inputs.map(\.shape),
            "output_shapes": ready.outputs.map(\.shape), "checked_elements": checked,
            "worker_exit_code": Int(status),
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
