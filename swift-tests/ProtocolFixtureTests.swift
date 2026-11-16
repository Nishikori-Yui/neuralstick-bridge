import Foundation
import XCTest
@testable import NCS2Kit

final class ProtocolFixtureTests: XCTestCase {
    func testReadyDescriptorsDecodeSharedCorpus() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("tests/worker_protocol_fixtures.json"))
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
}
