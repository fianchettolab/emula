import XCTest
@testable import Emula

/// The sampler on made-up logits. Its agreement with upstream on real positions is checked in
/// ParityTests.
final class SamplerTests: XCTestCase {
    // Probabilities at temperature 1: 0.5, 0.3, 0.15, 0.05 for indices 10, 20, 30, 40.
    let logits: [Float] = {
        var logits = [Float](repeating: 0, count: MaiaMoveDecoder.count)
        for (index, p) in [(10, 0.5), (20, 0.3), (30, 0.15), (40, 0.05)] {
            logits[index] = Float(log(p))
        }
        logits[50] = 100 // illegal, must never be picked
        return logits
    }()
    let legal = [10, 20, 30, 40]

    func testTemperatureZeroIsArgmaxWithLowestIndexOnTies() {
        let sampler = MaiaSampler(temperature: 0)
        XCTAssertEqual(sampler.candidates(logits: logits, legal: legal).map(\.index), [10])
        var tied = [Float](repeating: 0, count: MaiaMoveDecoder.count)
        tied[7] = 2; tied[3] = 2; tied[9] = 1
        XCTAssertEqual(sampler.candidates(logits: tied, legal: [9, 7, 3]).map(\.index), [3])
    }

    func testTopPExcludesTheMoveThatCrossesTheThreshold() {
        // Cumulative 0.5, 0.8, 0.95, 1.0. Upstream keeps moves while cumulative <= p.
        let cases: [(Double, [Int])] = [(0.5, [10]), (0.79, [10]), (0.81, [10, 20]), (0.9, [10, 20]), (0.96, [10, 20, 30]), (1, [10, 20, 30, 40])]
        for (topP, expected) in cases {
            let kept = MaiaSampler(temperature: 1, topP: topP).candidates(logits: logits, legal: legal)
            XCTAssertEqual(kept.map(\.index), expected, "top-p \(topP)")
            XCTAssertEqual(kept.reduce(0) { $0 + $1.probability }, 1, accuracy: 1e-12)
        }
        // The top move is kept even when it alone exceeds p.
        XCTAssertEqual(MaiaSampler(temperature: 1, topP: 0.1).candidates(logits: logits, legal: legal).map(\.index), [10])
    }

    func testTemperatureSharpensAndFlattens() {
        let sharp = MaiaSampler(temperature: 0.5).candidates(logits: logits, legal: legal)
        let flat = MaiaSampler(temperature: 2).candidates(logits: logits, legal: legal)
        XCTAssertGreaterThan(sharp[0].probability, 0.5)
        XCTAssertLessThan(flat[0].probability, 0.5)
        XCTAssertEqual(MaiaSampler(temperature: 1).candidates(logits: logits, legal: legal)[0].probability, 0.5, accuracy: 1e-6)
    }

    func testSamplingIsReproducibleAndFollowsTheDistribution() {
        let sampler = MaiaSampler(temperature: 1)
        func draw(seed: UInt64) -> [Int] {
            var generator = MaiaRandom(seed: seed)
            return (0..<20_000).map { _ in sampler.choose(logits: logits, legal: legal, using: &generator)! }
        }
        let first = draw(seed: 42)
        XCTAssertEqual(first, draw(seed: 42))
        XCTAssertNotEqual(first, draw(seed: 43))
        XCTAssertTrue(first.allSatisfy(legal.contains))
        for (index, p) in [(10, 0.5), (20, 0.3), (30, 0.15), (40, 0.05)] {
            let share = Double(first.filter { $0 == index }.count) / Double(first.count)
            XCTAssertEqual(share, p, accuracy: 0.015, "index \(index)")
        }
    }

    func testNoLegalMovesGivesNothing() {
        var generator = MaiaRandom(seed: 1)
        XCTAssertNil(MaiaSampler().choose(logits: logits, legal: [], using: &generator))
    }

    func testWDLFromValueLogits() {
        let wdl = MaiaWDL(valueLogits: [Float(log(0.2)), Float(log(0.1)), Float(log(0.7))])
        XCTAssertEqual(wdl.loss, 0.2, accuracy: 1e-6)
        XCTAssertEqual(wdl.draw, 0.1, accuracy: 1e-6)
        XCTAssertEqual(wdl.win, 0.7, accuracy: 1e-6)
    }
}
