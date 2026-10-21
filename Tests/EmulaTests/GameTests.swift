import XCTest
@testable import Emula

/// bestMove end to end: every move legal, and a fixed seed replays the same game.
final class GameTests: XCTestCase {
    func testOpeningMoveAtTemperatureZero() async throws {
        let session = try await Sessions.shared.session("5m")
        let result = try await session.bestMove(in: MaiaGame(), selfElo: 1500, opponentElo: 1500, temperature: 0)
        XCTAssertEqual(result.move, "e2e4")
        XCTAssertEqual(result.topMoves.count, 10)
        XCTAssertEqual(result.topMoves[0].move, "e2e4")
        XCTAssertEqual(result.wdl.win + result.wdl.draw + result.wdl.loss, 1, accuracy: 1e-9)
    }

    /// Every fixture position, sampled at temperature 1 in both history modes: the move must
    /// always be legal, including castling, en passant and (under)promotions.
    func testEveryFixturePositionGivesALegalMove() async throws {
        let session = try await Sessions.shared.session("5m")
        await session.reseed(7)
        for c in Fixture.m5.cases {
            let legal = Set(try c.game.legalMoves())
            for topP in [1.0, 0.9] {
                let result = try await session.bestMove(
                    in: c.game, selfElo: c.selfElo, opponentElo: c.opponentElo,
                    temperature: 1, topP: topP, historyMode: c.mode
                )
                XCTAssertTrue(legal.contains(result.move), "\(c.id) \(c.label): \(result.move)")
                XCTAssertTrue(result.topMoves.allSatisfy { legal.contains($0.move) })
            }
        }
    }

    func testSeededGamesReplayExactlyAndStayLegal() async throws {
        let first = try await playGame(seed: 2026)
        let second = try await playGame(seed: 2026)
        let other = try await playGame(seed: 2027)
        XCTAssertEqual(first, second)
        XCTAssertNotEqual(first, other)
        XCTAssertGreaterThan(first.count, 20)
    }

    /// A self-play game at temperature 1, until no legal move is left or 160 plies. Each
    /// move is checked against the legal moves before it is played.
    func playGame(seed: UInt64) async throws -> [String] {
        let session = try await Sessions.shared.session("5m")
        await session.reseed(seed)
        var game = MaiaGame()
        while game.moves.count < 160 {
            let legal = try game.legalMoves()
            if legal.isEmpty {
                do {
                    _ = try await session.bestMove(in: game, selfElo: 1500, opponentElo: 1500, temperature: 1)
                    XCTFail("expected noLegalMoves")
                } catch {
                    XCTAssertEqual(error as? MaiaError, .noLegalMoves)
                }
                break
            }
            let elo = game.moves.count.isMultiple(of: 2) ? (1200, 1800) : (1800, 1200)
            let result = try await session.bestMove(
                in: game, selfElo: elo.0, opponentElo: elo.1, temperature: 1, topP: 0.95, historyMode: .recentPositions
            )
            XCTAssertTrue(legal.contains(result.move), "ply \(game.moves.count): \(result.move)")
            game.moves.append(result.move)
        }
        return game.moves
    }

    func testCheckmateHasNoMove() async throws {
        let session = try await Sessions.shared.session("5m")
        let mated = MaiaGame(moves: ["f2f3", "e7e5", "g2g4", "d8h4"])
        do {
            _ = try await session.bestMove(in: mated, selfElo: 1500, opponentElo: 1500, temperature: 0)
            XCTFail("expected noLegalMoves")
        } catch {
            XCTAssertEqual(error as? MaiaError, .noLegalMoves)
        }
    }
}
