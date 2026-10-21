import CoreML
import Foundation

/// Raw model outputs for one position (spec section 2).
public struct MaiaOutput: Sendable {
    /// One logit per vocabulary entry, in the side-to-move frame.
    public var moveLogits: [Float]
    /// `[loss, draw, win]` logits for the side to move.
    public var valueLogits: [Float]
    public var ponder: Float
}

/// Win, draw and loss probabilities for the side to move.
public struct MaiaWDL: Sendable, Hashable {
    public var win: Double
    public var draw: Double
    public var loss: Double

    /// From the value head's `[loss, draw, win]` logits (spec section 6).
    public init(valueLogits: [Float]) {
        let weights = valueLogits.map { exp(Double($0) - Double(valueLogits.max()!)) }
        let total = weights.reduce(0, +)
        (loss, draw, win) = (weights[0] / total, weights[1] / total, weights[2] / total)
    }
}

public struct MaiaMoveProbability: Sendable, Hashable {
    /// UCI, e.g. `e2e4`, `e1g1`, `e7e8q`.
    public var move: String
    public var probability: Double
}

public struct MaiaResult: Sendable {
    /// The chosen move in UCI. Always legal in the given position.
    public var move: String
    /// The most likely moves at temperature 1, most likely first.
    public var topMoves: [MaiaMoveProbability]
    /// The current position's outlook for the side to move.
    public var wdl: MaiaWDL
}

/// Loads a Maia-3 Core ML model once and runs it off the main thread.
///
/// Inference never touches the network: the model is a local file.
public actor MaiaSession {
    public nonisolated let computeUnits: MLComputeUnits
    private let model: MLModel
    private var generator: MaiaRandom

    /// - Parameters:
    ///   - modelURL: a compiled `.mlmodelc` (what Xcode puts in an app bundle) or an
    ///     `.mlpackage`, which is compiled first.
    ///   - computeUnits: preferred compute units. If the model fails to load or run with
    ///     them, the session falls back to the CPU.
    ///   - seed: seeds the move sampler, so a sequence of `bestMove` calls is reproducible.
    public init(modelURL: URL, computeUnits: MLComputeUnits = .all, seed: UInt64 = .random(in: .min ... .max)) async throws {
        let compiledURL = modelURL.pathExtension == "mlmodelc" ? modelURL : try await MLModel.compileModel(at: modelURL)
        (model, self.computeUnits) = try await Self.load(compiledURL, preferring: computeUnits)
        generator = MaiaRandom(seed: seed)
    }

    /// Loads and warms up the model with the preferred compute units, then with the CPU only.
    /// Throws the last error if neither works.
    private static func load(_ url: URL, preferring units: MLComputeUnits) async throws -> (MLModel, MLComputeUnits) {
        var lastError: Error?
        for candidate in units == .cpuOnly ? [.cpuOnly] : [units, .cpuOnly] {
            do {
                let configuration = MLModelConfiguration()
                configuration.computeUnits = candidate
                let model = try await MLModel.load(contentsOf: url, configuration: configuration)
                _ = try run(model, tokens: warmUpTokens, selfElo: 1500, opponentElo: 1500)
                return (model, candidate)
            } catch {
                lastError = error
            }
        }
        throw lastError!
    }

    /// Restarts the move sampler's random sequence.
    public func reseed(_ seed: UInt64) {
        generator = MaiaRandom(seed: seed)
    }

    /// Raw outputs for a prepared input (spec section 2).
    public func predict(tokens: [Float], selfElo: Int, opponentElo: Int) async throws -> MaiaOutput {
        try Self.run(model, tokens: tokens, selfElo: selfElo, opponentElo: opponentElo)
    }

    /// Picks Maia's move in the game's current position.
    ///
    /// - Parameters:
    ///   - selfElo: rating Maia plays at (the side to move).
    ///   - opponentElo: rating of the opponent.
    ///   - temperature: 0 always plays the most likely move; 1 samples the model's
    ///     distribution as it is.
    ///   - topP: only sample among the most likely moves covering this much probability
    ///     (upstream's rule, spec 8.5); 1 disables it.
    ///   - topCount: how many of the most likely moves to report.
    public func bestMove(
        in game: MaiaGame,
        selfElo: Int,
        opponentElo: Int,
        temperature: Double,
        topP: Double = 1,
        historyMode: MaiaHistoryMode = .currentPosition,
        topCount: Int = 10
    ) async throws -> MaiaResult {
        let (positions, legalMoves) = try game.replay()
        let blackToMove = positions.last!.turn == .black
        let legal = legalMoves.compactMap { MaiaMoveDecoder.index(of: $0, blackToMove: blackToMove) }
        guard !legal.isEmpty else { throw MaiaError.noLegalMoves }

        let tokens = MaiaEncoder.tokens(for: positions, mode: historyMode)
        let output = try await predict(tokens: tokens, selfElo: selfElo, opponentElo: opponentElo)

        let sampler = MaiaSampler(temperature: temperature, topP: topP)
        let chosen = sampler.choose(logits: output.moveLogits, legal: legal, using: &generator)!
        let top = MaiaSampler.softmax(logits: output.moveLogits, legal: legal).prefix(topCount)

        return MaiaResult(
            move: MaiaMoveDecoder.uci(at: chosen, blackToMove: blackToMove)!,
            topMoves: top.map { MaiaMoveProbability(move: MaiaMoveDecoder.uci(at: $0.index, blackToMove: blackToMove)!, probability: $0.probability) },
            wdl: MaiaWDL(valueLogits: output.valueLogits)
        )
    }

    // MARK: Core ML

    // The start position always encodes.
    private static let warmUpTokens = try! MaiaEncoder.tokens(for: MaiaGame(), mode: .currentPosition)

    /// Synchronous on purpose: MLModel isn't Sendable, so it runs on the actor's executor
    /// (already off the main thread) rather than being handed to Core ML's async API.
    private static func run(_ model: MLModel, tokens: [Float], selfElo: Int, opponentElo: Int) throws -> MaiaOutput {
        let elo = { (value: Int) in MLMultiArray(MLShapedArray<Int32>(scalars: [Int32(min(max(value, 0), 5000))], shape: [1])) }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLMultiArray(MLShapedArray<Float>(scalars: tokens, shape: [1, 64, MaiaEncoder.columns])),
            "self_elo": elo(selfElo),
            "opponent_elo": elo(opponentElo),
        ])
        let outputs = try model.prediction(from: inputs)
        func scalars(_ name: String, count: Int) throws -> [Float] {
            guard let array = outputs.featureValue(for: name)?.multiArrayValue, array.count == count else {
                throw MaiaError.modelOutput(name)
            }
            return MLShapedArray<Float>(converting: array).scalars
        }
        return MaiaOutput(
            moveLogits: try scalars("move_logits", count: MaiaMoveDecoder.count),
            valueLogits: try scalars("value_logits", count: 3),
            ponder: try scalars("ponder", count: 1)[0]
        )
    }
}
