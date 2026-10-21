import XCTest
@testable import Emula

/// The encoder and Scacchi's move generation against the Python fixtures. No model needed.
final class EncoderTests: XCTestCase {
    func testEncodingMatchesFixturesExactly() throws {
        for fixture in [Fixture.m23, Fixture.m5] {
            for c in fixture.cases {
                let tokens = try MaiaEncoder.tokens(for: c.game, mode: c.mode)
                XCTAssertEqual(tokens.count, 64 * 97)
                let active = tokens.indices.filter { tokens[$0] != 0 }
                XCTAssertEqual(active, c.tokensActive, "\(fixture.meta.model) \(c.id) \(c.label)")
                XCTAssertTrue(tokens.allSatisfy { $0 == 0 || $0 == 1 })
            }
        }
    }

    /// Maia's legality rests on Scacchi's (shakmaty's) move generator: it must agree with python-chess.
    func testLegalMovesMatchPythonChess() throws {
        for c in Fixture.m23.cases {
            let legal = Set(try c.game.legalMoves())
            XCTAssertEqual(legal.count, c.legalMoveCount, "\(c.id) \(c.label) \(c.fen)")
            for move in c.topMoves {
                XCTAssertTrue(legal.contains(move.uci), "\(c.id): \(move.uci) not legal per Scacchi")
                XCTAssertEqual(MaiaMoveDecoder.index(of: move.uci, blackToMove: c.blackToMove), move.index, "\(c.id) \(move.uci)")
            }
        }
    }

    func testEmptyHistoryAtStartIsTheSameInBothModes() throws {
        let current = try MaiaEncoder.tokens(for: MaiaGame(), mode: .currentPosition)
        let recent = try MaiaEncoder.tokens(for: MaiaGame(), mode: .recentPositions)
        XCTAssertEqual(current, recent)
        // a1 holds White's rook (channel 3) and e1 White's king (channel 5), in every frame.
        for frame in 0..<8 {
            XCTAssertEqual(current[0 * 97 + frame * 12 + 3], 1)
            XCTAssertEqual(current[4 * 97 + frame * 12 + 5], 1)
        }
        XCTAssertEqual(current.reduce(0, +), 32 * 8)
        XCTAssertTrue((0..<64).allSatisfy { current[$0 * 97 + 96] == 0 })
    }

    func testBlackToMoveIsMirrored() throws {
        // After 1.e4, Black's king sits on e1 (square 4) and Black's e-pawn on e2 (square 12)
        // as the side to move. White's e-pawn appears on e5 (square 36) as an opponent pawn
        // (channel 6), and its starting square, mirrored to e7 (square 52), is empty.
        let tokens = try MaiaEncoder.tokens(for: MaiaGame(moves: ["e2e4"]), mode: .currentPosition)
        let current = 7 * 12
        XCTAssertEqual(tokens[4 * 97 + current + 5], 1)
        XCTAssertEqual(tokens[12 * 97 + current + 0], 1)
        XCTAssertEqual(tokens[36 * 97 + current + 6], 1)
        XCTAssertEqual(tokens[52 * 97 + current + 6], 0)
    }

    func testRecentPositionsPadsWithTheEarliestFrame() throws {
        let tokens = try MaiaEncoder.tokens(for: MaiaGame(moves: ["e2e4"]), mode: .recentPositions)
        let start = try MaiaEncoder.tokens(for: MaiaGame(), mode: .currentPosition)
        // Frames 0-6 are the start position (White to move), frame 7 the position after e4.
        for square in 0..<64 {
            for column in 0..<(7 * 12) {
                XCTAssertEqual(tokens[square * 97 + column], start[square * 97 + column])
            }
        }
    }

    func testRejectsBadInput() {
        XCTAssertThrowsError(try MaiaGame(moves: ["e2e5"]).legalMoves()) {
            XCTAssertEqual($0 as? MaiaError, .illegalMove("e2e5", ply: 0))
        }
        for malformed in ["", "e2", "e2e4x", "i2i4", "e9e4", "e7e8k", "E2E4"] {
            XCTAssertThrowsError(try MaiaGame(moves: [malformed]).legalMoves(), malformed)
        }
        XCTAssertThrowsError(try MaiaGame(startFEN: "not a fen").legalMoves()) {
            XCTAssertEqual($0 as? MaiaError, .invalidFEN("not a fen"))
        }
    }
}
