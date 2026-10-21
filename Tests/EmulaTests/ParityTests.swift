import XCTest
@testable import Emula

/// The Core ML models through Emula against the PyTorch fixtures: top-1 move, top-10
/// probabilities and W/D/L within 1e-3, and the sampler's choices.
final class ParityTests: XCTestCase {
    static let tolerance = 1e-3

    func test23MAllComputeUnits() async throws {
        try await checkParity(size: "23m", fixture: .m23, computeUnits: .all)
    }

    func test5MAllComputeUnits() async throws {
        try await checkParity(size: "5m", fixture: .m5, computeUnits: .all)
    }

    /// The CPU fallback must give the same answers.
    func test5MCPUOnly() async throws {
        try await checkParity(size: "5m", fixture: .m5, computeUnits: .cpuOnly)
    }

    func checkParity(size: String, fixture: Fixture, computeUnits: MLComputeUnitsChoice) async throws {
        let session = try await Sessions.shared.session(size, computeUnits: computeUnits)
        var worstProb = 0.0, worstValue = 0.0, boundaryCases = 0

        for c in fixture.cases {
            let context = "\(size) \(computeUnits) \(c.id) \(c.label)"
            let tokens = try MaiaEncoder.tokens(for: c.game, mode: c.mode)
            let output = try await session.predict(tokens: tokens, selfElo: c.selfElo, opponentElo: c.opponentElo)
            let legal = try c.game.legalMoves().map { MaiaMoveDecoder.index(of: $0, blackToMove: c.blackToMove)! }
            let probabilities = MaiaSampler.softmax(logits: output.moveLogits, legal: legal)
            let probability = Dictionary(uniqueKeysWithValues: probabilities.map { ($0.index, $0.probability) })

            if !c.nearTie {
                XCTAssertEqual(probabilities[0].index, c.topMoves[0].index, "top-1: \(context)")
            }
            for move in c.topMoves {
                let error = abs(probability[move.index]! - move.prob)
                worstProb = max(worstProb, error)
                XCTAssertLessThanOrEqual(error, Self.tolerance, "\(move.uci): \(context)")
            }

            let wdl = MaiaWDL(valueLogits: output.valueLogits)
            for (ours, theirs) in [(wdl.win, c.value.win), (wdl.draw, c.value.draw), (wdl.loss, c.value.loss)] {
                worstValue = max(worstValue, abs(ours - theirs))
                XCTAssertEqual(ours, theirs, accuracy: Self.tolerance, "W/D/L: \(context)")
            }

            for expected in c.sampling {
                let sampler = MaiaSampler(temperature: expected.temperature, topP: expected.topP)
                let ours = sampler.candidates(logits: output.moveLogits, legal: legal)
                    .map { (uci: MaiaMoveDecoder.uci(at: $0.index, blackToMove: c.blackToMove)!, probability: $0.probability) }
                if let argmax = expected.argmax {
                    XCTAssertEqual(ours.map(\.uci), [argmax], "argmax: \(context)")
                    continue
                }
                let theirs = expected.nucleus!
                if ours.map(\.uci) != theirs.map(\.uci) {
                    // Allowed only where a move sits right on the top-p threshold, so float
                    // differences between the runtimes decide whether it's kept.
                    let all = MaiaSampler(temperature: expected.temperature).candidates(logits: output.moveLogits, legal: legal)
                    let shorter = min(ours.count, theirs.count)
                    let edge = all.prefix(shorter + 1).reduce(0) { $0 + $1.probability }
                    let onBoundary = abs(ours.count - theirs.count) == 1
                        && ours.prefix(shorter).map(\.uci) == theirs.prefix(shorter).map(\.uci)
                        && abs(edge - expected.topP) < 1e-4
                    XCTAssertTrue(onBoundary, "nucleus T=\(expected.temperature) p=\(expected.topP): \(context)")
                    boundaryCases += 1
                    continue
                }
                for (a, b) in zip(ours, theirs) {
                    XCTAssertEqual(a.probability, b.prob, accuracy: Self.tolerance, "nucleus \(a.uci): \(context)")
                }
            }
        }
        print("\(size) \(computeUnits) [\(session.computeUnits.rawValue)]: \(fixture.cases.count) cases, "
            + "worst top-10 prob error \(worstProb), worst W/D/L error \(worstValue), boundary nuclei \(boundaryCases)")
    }
}
