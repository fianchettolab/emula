import CoreML
import Foundation
import XCTest
@testable import Emula

/// Paths into the repo. Fixtures are committed; exported models live in the untracked
/// Tools/maia3_export/out (built by export_maia3.py), so tests that need one skip without it.
enum Repo {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let export = root.appendingPathComponent("Tools/maia3_export")

    static func model(_ size: String) throws -> URL {
        let url = export.appendingPathComponent("out/maia3-\(size)-fp32.mlpackage")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("\(url.lastPathComponent) not found; run Tools/maia3_export/export_maia3.py")
        }
        return url
    }
}

/// One session per model for the whole test run: compiling and loading takes a second or two.
actor Sessions {
    static let shared = Sessions()
    private var sessions: [String: MaiaSession] = [:]

    func session(_ size: String, computeUnits: MLComputeUnitsChoice = .all) async throws -> MaiaSession {
        let key = "\(size)-\(computeUnits)"
        if let session = sessions[key] { return session }
        let session = try await MaiaSession(modelURL: Repo.model(size), computeUnits: computeUnits.units, seed: 0)
        sessions[key] = session
        return session
    }
}

enum MLComputeUnitsChoice: String, Sendable {
    case all, cpuOnly
    var units: MLComputeUnits { self == .all ? .all : .cpuOnly }
}

// MARK: Fixtures (format: Tools/maia3_export/fixtures/README.md)

struct Fixture: Decodable, Sendable {
    struct Meta: Decodable, Sendable {
        let model: String
        let cases: Int
    }

    struct Case: Decodable, Sendable {
        struct TopMove: Decodable, Sendable {
            let uci: String
            let index: Int
            let logit: Double
            let prob: Double
        }

        struct Value: Decodable, Sendable {
            let loss: Double
            let draw: Double
            let win: Double
        }

        struct Sampling: Decodable, Sendable {
            struct Kept: Decodable, Sendable {
                let uci: String
                let prob: Double
            }

            let temperature: Double
            let topP: Double
            let argmax: String?
            let nucleus: [Kept]?

            enum CodingKeys: String, CodingKey {
                case temperature, argmax, nucleus
                case topP = "top_p"
            }
        }

        let id: String
        let label: String
        let tags: [String]
        let startFen: String
        let moves: [String]
        let fen: String
        let historyMode: String
        let selfElo: Int
        let opponentElo: Int
        let tokensActive: [Int]
        let legalMoveCount: Int
        let topMoves: [TopMove]
        let nearTie: Bool
        let value: Value
        let sampling: [Sampling]

        enum CodingKeys: String, CodingKey {
            case id, label, tags, moves, fen, value, sampling
            case startFen = "start_fen"
            case historyMode = "history_mode"
            case selfElo = "self_elo"
            case opponentElo = "opponent_elo"
            case tokensActive = "tokens_active"
            case legalMoveCount = "legal_move_count"
            case topMoves = "top_moves"
            case nearTie = "near_tie"
        }

        var game: MaiaGame { MaiaGame(startFEN: startFen, moves: moves) }
        var mode: MaiaHistoryMode { historyMode == "current" ? .currentPosition : .recentPositions }
        var blackToMove: Bool { fen.split(separator: " ")[1] == "b" }
    }

    let meta: Meta
    let cases: [Case]

    static let m23 = load("23m")
    static let m5 = load("5m")

    static func load(_ size: String) -> Fixture {
        let url = Repo.export.appendingPathComponent("fixtures/maia3-\(size)-fixtures.json")
        return try! JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }
}
