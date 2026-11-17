import Foundation

public enum NCS2ModelInspectionError: Error, Sendable, Equatable {
    case invalidArgument
    case launchFailed(String)
    case modelRejected(String)
    case invalidResponse
}

public struct NCS2TensorInspection: Sendable, Equatable, Decodable {
    public let index: Int
    public let name: String
    public let dtype: String
    public let isStatic: Bool
    public let rank: Int?
    public let shape: [Int]?
    public let elements: Int?
    public let elementLimitOK: Bool

    enum CodingKeys: String, CodingKey {
        case index, name, dtype, rank, shape, elements
        case isStatic = "static"
        case elementLimitOK = "element_limit_ok"
    }
}

public struct NCS2ModelCompatibility: Sendable, Equatable, Decodable {
    public let compatible: Bool
    public let reasons: [String]
}

public struct NCS2ModelInspection: Sendable, Equatable {
    public let xml: URL
    public let openVINOVersion: String
    public let inputs: [NCS2TensorInspection]
    public let outputs: [NCS2TensorInspection]
    public let schema1: NCS2ModelCompatibility
    public let schema2: NCS2ModelCompatibility
    public let hardwareAccessed: Bool
}

private struct InspectionEnvelope: Decodable {
    struct Model: Decodable {
        let xml: URL
        let inputs: [NCS2TensorInspection]
        let outputs: [NCS2TensorInspection]
        let inputCount: Int
        let outputCount: Int

        enum CodingKeys: String, CodingKey {
            case xml, inputs, outputs
            case inputCount = "input_count"
            case outputCount = "output_count"
        }
    }

    struct Compatibility: Decodable {
        let schema1: NCS2ModelCompatibility
        let schema2: NCS2ModelCompatibility

        enum CodingKeys: String, CodingKey {
            case schema1 = "schema_1"
            case schema2 = "schema_2"
        }
    }

    let schemaVersion: Int
    let type: String
    let hardwareAccessed: Bool
    let openVINOVersion: String
    let model: Model
    let compatibility: Compatibility

    enum CodingKeys: String, CodingKey {
        case type, model, compatibility
        case schemaVersion = "schema_version"
        case hardwareAccessed = "hardware_accessed"
        case openVINOVersion = "openvino_version"
    }
}

/// Reads OpenVINO IR metadata with the native inspector. This never opens USB.
public struct NCS2ModelInspector: Sendable {
    public let executable: URL

    public init(executable: URL) throws {
        guard executable.isFileURL else { throw NCS2ModelInspectionError.invalidArgument }
        self.executable = executable
    }

    public func inspect(xml: URL) async throws -> NCS2ModelInspection {
        guard xml.isFileURL else { throw NCS2ModelInspectionError.invalidArgument }
        try Task.checkCancellation()
        let executable = self.executable
        let output: Data = try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Data, any Error>) in
            DispatchQueue.global(qos: .utility).async {
                let process = Process()
                let stdout = Pipe()
                process.executableURL = executable
                process.arguments = ["--model", xml.standardizedFileURL.path]
                process.standardOutput = stdout
                process.standardError = FileHandle.standardError
                do {
                    try process.run()
                    let data = stdout.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()
                    guard process.terminationStatus == 0 else {
                        continuation.resume(throwing: NCS2ModelInspectionError.modelRejected(
                            "Native inspector exited with status \(process.terminationStatus)"))
                        return
                    }
                    continuation.resume(returning: data)
                } catch {
                    continuation.resume(throwing: NCS2ModelInspectionError.launchFailed(String(describing: error)))
                }
            }
        }
        try Task.checkCancellation()
        let envelope: InspectionEnvelope
        do {
            envelope = try JSONDecoder().decode(InspectionEnvelope.self, from: output)
        } catch {
            throw NCS2ModelInspectionError.invalidResponse
        }
        guard envelope.schemaVersion == 1, envelope.type == "model_inspection",
              envelope.hardwareAccessed == false,
              envelope.model.inputCount == envelope.model.inputs.count,
              envelope.model.outputCount == envelope.model.outputs.count else {
            throw NCS2ModelInspectionError.invalidResponse
        }
        return NCS2ModelInspection(
            xml: envelope.model.xml,
            openVINOVersion: envelope.openVINOVersion,
            inputs: envelope.model.inputs,
            outputs: envelope.model.outputs,
            schema1: envelope.compatibility.schema1,
            schema2: envelope.compatibility.schema2,
            hardwareAccessed: envelope.hardwareAccessed
        )
    }
}
