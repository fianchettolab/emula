import XCTest
@testable import Emula

final class DecoderTests: XCTestCase {
    func testRoundTripsEveryIndexForBothSides() {
        for blackToMove in [false, true] {
            for index in 0..<MaiaMoveDecoder.count {
                let uci = MaiaMoveDecoder.uci(at: index, blackToMove: blackToMove)!
                XCTAssertEqual(MaiaMoveDecoder.index(of: uci, blackToMove: blackToMove), index, uci)
            }
        }
        XCTAssertNil(MaiaMoveDecoder.uci(at: -1, blackToMove: false))
        XCTAssertNil(MaiaMoveDecoder.uci(at: MaiaMoveDecoder.count, blackToMove: false))
    }

    /// The worked examples in docs/maia3-spec.md section 5.
    func testSpecExamples() {
        let examples: [(String, Bool, Int)] = [
            ("e2e4", false, 796), ("g1f3", false, 405), ("e7e5", true, 796),
            ("a7a8q", false, 4096), ("a7a8n", false, 4099), ("h7h8n", false, 4351),
            ("b2b1q", true, 4132), ("b2b1n", true, 4135),
            ("e1g1", false, 262), ("e1c1", false, 258), ("e8g8", true, 262), ("e8c8", true, 258),
            ("e5d6", false, 2347), ("a7a8", false, 3128),
        ]
        for (uci, black, index) in examples {
            XCTAssertEqual(MaiaMoveDecoder.index(of: uci, blackToMove: black), index, uci)
            XCTAssertEqual(MaiaMoveDecoder.uci(at: index, blackToMove: black), uci, uci)
        }
    }

    func testPromotionsOnlyFromTheSeventhRank() {
        XCTAssertNil(MaiaMoveDecoder.index(of: "a6a7q", blackToMove: false))
        XCTAssertNil(MaiaMoveDecoder.index(of: "a2a1q", blackToMove: false), "White can't promote on rank 1")
        XCTAssertNil(MaiaMoveDecoder.index(of: "a7a8q", blackToMove: true), "Black promotes on rank 1")
        XCTAssertNil(MaiaMoveDecoder.index(of: "a7a8k", blackToMove: false))
        // Capture-promotions, all four pieces.
        XCTAssertEqual(MaiaMoveDecoder.index(of: "c7b8r", blackToMove: false), 4096 + (2 * 8 + 1) * 4 + 1)
        XCTAssertEqual(MaiaMoveDecoder.index(of: "c2d1b", blackToMove: true), 4096 + (2 * 8 + 3) * 4 + 2)
    }

    func testRejectsMalformedMoves() {
        for uci in ["", "e2", "e2e", "e2e4qq", "z2e4", "e0e4", "e2e9"] {
            XCTAssertNil(MaiaMoveDecoder.index(of: uci, blackToMove: false), uci)
        }
    }
}
