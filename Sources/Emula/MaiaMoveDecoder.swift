/// Maps between UCI moves and indices of the model's 4352 move logits (spec section 5).
///
/// Indices live in the side-to-move frame: Black's moves are mirrored (rank r → 9 − r)
/// before lookup and after decoding.
public enum MaiaMoveDecoder {
    public static let count = 4352
    static let promotionBase = 4096
    static let files = Array("abcdefgh")
    static let promotionPieces = Array("qrbn")

    public static func index(of uci: String, blackToMove: Bool) -> Int? {
        let chars = Array(blackToMove ? mirror(uci) : uci)
        guard chars.count == 4 || chars.count == 5,
              let from = square(chars[0], chars[1]), let to = square(chars[2], chars[3]) else {
            return nil
        }
        guard chars.count == 5 else { return from * 64 + to }
        guard from / 8 == 6, to / 8 == 7, let piece = promotionPieces.firstIndex(of: chars[4]) else {
            return nil
        }
        return promotionBase + ((from % 8) * 8 + to % 8) * 4 + piece
    }

    public static func uci(at index: Int, blackToMove: Bool) -> String? {
        let uci: String
        switch index {
        case 0..<promotionBase:
            uci = name(index / 64) + name(index % 64)
        case promotionBase..<count:
            let offset = index - promotionBase
            let fromFile = offset / 32, toFile = offset / 4 % 8, piece = offset % 4
            uci = "\(files[fromFile])7\(files[toFile])8\(promotionPieces[piece])"
        default:
            return nil
        }
        return blackToMove ? mirror(uci) : uci
    }

    static func mirror(_ uci: String) -> String {
        String(uci.map { $0.wholeNumberValue.map { Character(String(9 - $0)) } ?? $0 })
    }

    static func square(_ file: Character, _ rank: Character) -> Int? {
        guard let f = files.firstIndex(of: file), let r = rank.wholeNumberValue, (1...8).contains(r) else {
            return nil
        }
        return (r - 1) * 8 + f
    }

    static func name(_ square: Int) -> String {
        "\(files[square % 8])\(square / 8 + 1)"
    }
}
