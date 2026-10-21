import Scacchi

/// A game as Maia sees it: the position it started from and the moves played since, in UCI
/// (`e2e4`, `e1g1` for castling, `e7e8q` for promotion).
public struct MaiaGame: Sendable, Hashable {
    public static let standardStartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    public var startFEN: String
    public var moves: [String]

    public init(startFEN: String = MaiaGame.standardStartFEN, moves: [String] = []) {
        self.startFEN = startFEN
        self.moves = moves
    }
}

public enum MaiaError: Error, Equatable {
    case invalidFEN(String)
    case illegalMove(String, ply: Int)
    case noLegalMoves
    case modelOutput(String)
}

extension MaiaGame {
    /// The game replayed with Scacchi: every position, oldest first, ending with the
    /// position to move in, and that position's legal moves in UCI. Each move is checked
    /// for legality.
    func replay() throws -> (positions: [BoardState], legalMoves: [String]) {
        let game: Game
        do {
            game = try Game.fromFen(fen: startFEN)
        } catch {
            throw MaiaError.invalidFEN(startFEN)
        }
        var positions = [game.board()]
        for (ply, uci) in moves.enumerated() {
            do {
                _ = try game.play(uci: uci)
            } catch {
                throw MaiaError.illegalMove(uci, ply: ply)
            }
            positions.append(game.board())
        }
        return (positions, game.legalMoves())
    }

    /// Every position of the game, oldest first, ending with the position to move in.
    func positions() throws -> [BoardState] {
        try replay().positions
    }

    /// The legal moves in the game's current position, in UCI.
    public func legalMoves() throws -> [String] {
        try replay().legalMoves
    }
}
