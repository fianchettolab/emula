import CoreML
import XCTest
@testable import Emula

/// `maia3Commit` against the export's manifest and the models' Core ML metadata, so the
/// code and the models it runs come from the same Maia-3.
final class UpstreamTests: XCTestCase {
    func testManifest() throws {
        let url = Repo.export.appendingPathComponent("manifest.json")
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        XCTAssertEqual(manifest?["maia3_commit"] as? String, maia3Commit)
    }

    func testModels() async throws {
        for size in ["5m", "23m"] {
            let compiled = try await MLModel.compileModel(at: Repo.model(size))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let metadata = try MLModel(contentsOf: compiled).modelDescription.metadata
            let defined = metadata[.creatorDefinedKey] as? [String: String]
            XCTAssertEqual(defined?["maia3_commit"], maia3Commit, size)
        }
    }
}
