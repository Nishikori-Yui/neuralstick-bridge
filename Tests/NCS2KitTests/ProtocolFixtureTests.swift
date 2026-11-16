import Foundation
import XCTest
@testable import NCS2Kit

final class ProtocolFixtureTests: XCTestCase {
    func testReadyDescriptorsDecodeSharedCorpus() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("python/tests/worker_protocol_fixtures.json"))
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let cases = try XCTUnwrap(corpus["cases"] as? [[String: Any]])
        for version in [1, 2] {
            let item = try XCTUnwrap(cases.first { $0["name"] as? String == "schema\(version)-ready" })
            let frame = try JSONSerialization.data(withJSONObject: XCTUnwrap(item["frame"]))
            if version == 1 {
                let ready = try JSONDecoder().decode(NCS2Ready.self, from: frame)
                XCTAssertEqual(try ready.input.elementCount(), 4)
                XCTAssertEqual(ready.device, "MYRIAD")
            } else {
                let ready = try JSONDecoder().decode(NCS2MultiReady.self, from: frame)
                XCTAssertEqual(ready.inputs.count, 2)
                XCTAssertEqual(try ready.inputs[0].elementCount(), 4)
            }
        }
    }
    func testSynchronousClientsUseSharedGoldenFrames() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let corpus = root.appendingPathComponent("python/tests/worker_protocol_fixtures.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes
        let literal = String(decoding: try encoder.encode(corpus.path), as: UTF8.self)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for version in [1, 2] {
            let source = #"""
#!/usr/bin/env python3
import json,sys
frames={c['name']:c['frame'] for c in json.load(open(CORPUS))['cases']}
prefix='schemaVERSION-'
print(json.dumps(frames[prefix+'ready']),flush=True)
for line in sys.stdin:
    req=json.loads(line)
    key='infer-response' if req['operation']=='infer' else 'control-response'
    response=dict(frames[prefix+key],request_id=req['request_id'])
    print(json.dumps(response),flush=True)
    if req['operation']=='shutdown': break
"""#.replacingOccurrences(of: "CORPUS", with: literal)
     .replacingOccurrences(of: "VERSION", with: String(version))
            let worker = directory.appendingPathComponent("worker")
            try Data(source.utf8).write(to: worker)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: worker.path)
            let model = directory.appendingPathComponent("fixture.xml")
            if version == 1 {
                let client = try NCS2WorkerClient(worker: worker, plugins: model, model: model)
                defer { try? client.close() }
                for _ in 0..<2 {
                    XCTAssertEqual(try client.infer([-2,0.5,-1,3]).output, [0,0.5,0,3])
                }
                _ = try client.shutdown()
            } else {
                let client = try NCS2MultiWorkerClient(worker: worker, plugins: model, model: model)
                defer { try? client.close() }
                for _ in 0..<2 {
                    XCTAssertEqual(try client.infer([[-2,0.5,-1,3], [-2,0.5,-1,3]])[0].data, [0,0.5,0,3])
                }
                _ = try client.shutdown()
            }
        }
    }

}
