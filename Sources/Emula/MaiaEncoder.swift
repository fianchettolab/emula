import Scacchi

/// Which past positions fill the model's eight history frames (spec 3.3).
public enum MaiaHistoryMode: String, Sendable, CaseIterable {
    /// The current position in all eight frames: the official engine's default.
    case currentPosition
    /// The last eight positions of the game, oldest first.
    case recentPositions
}

/// Turns a game into the model's `tokens` input, `[64, 97]` row-major (spec section 3).
public enum MaiaEncoder {
    public static let frameCount = 8
    public static let columns = 12 * frameCount + 1
    public static let tokenCount = 64 * columns

    public static func tokens(for game: MaiaGame, mode: MaiaHistoryMode) throws -> [Float] {
        tokens(for: try game.positions(), mode: mode)
    }

    /// `positions` runs oldest first; the last one is the position to move in.
    static func tokens(for positions: [BoardState], mode: MaiaHistoryMode) -> [Float] {
        var frames = mode == .currentPosition ? [positions.last!] : Array(positions.suffix(frameCount))
        frames.insert(contentsOf: repeatElement(frames[0], count: frameCount - frames.count), at: 0)

        var tokens = [Float](repeating: 0, count: tokenCount)
        for (frame, position) in frames.enumerated() {
            for (square, channel) in planes(of: position) {
                tokens[square * columns + 12 * frame + channel] = 1
            }
        }
        return tokens
    }

    /// One frame (spec 3.1): squares a1 = 0 … h8 = 63, mirrored when Black is to move;
    /// channels 0–5 are the side to move's P N B R Q K, 6–11 the opponent's.
    static func planes(of position: BoardState) -> [(square: Int, channel: Int)] {
        let mover = position.turn
        return position.pieces.map { piece in
            // Scacchi's squares are rank-first already (a1 = 0, b1 = 1), like Maia's.
            let square = Int(piece.square)
            let rank = mover == .black ? 7 - square / 8 : square / 8
            let channel = plane(of: piece.role) + (piece.color == mover ? 0 : 6)
            return (rank * 8 + square % 8, channel)
        }
    }

    static func plane(of kind: Role) -> Int {
        switch kind {
        case .pawn: 0
        case .knight: 1
        case .bishop: 2
        case .rook: 3
        case .queen: 4
        case .king: 5
        }
    }
}
