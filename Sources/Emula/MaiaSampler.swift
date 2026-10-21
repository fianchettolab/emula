import Foundation

/// Picks a move from the model's logits the way the official engine does (spec section 8):
/// illegal moves are excluded, temperature 0 means argmax, otherwise softmax at the given
/// temperature, restricted by top-p.
public struct MaiaSampler: Sendable, Hashable {
    public var temperature: Double
    public var topP: Double

    public init(temperature: Double = 1, topP: Double = 1) {
        self.temperature = temperature
        self.topP = topP
    }

    /// The moves the sampler can pick, with their probabilities, most likely first (ties by
    /// index). Upstream's top-p rule: keep moves while the cumulative probability is ≤ topP,
    /// always keep the most likely move. Temperature ≤ 0 gives the argmax alone.
    public func candidates(logits: [Float], legal: [Int]) -> [(index: Int, probability: Double)] {
        guard !legal.isEmpty else { return [] }
        guard temperature > 0 else {
            // Upstream torch.argmax: the lowest index wins a tie.
            let best = legal.min { (logits[$0], -$0) > (logits[$1], -$1) }!
            return [(best, 1)]
        }
        let sorted = Self.softmax(logits: logits, legal: legal, temperature: temperature)
        guard topP < 1 else { return sorted }

        var kept = [sorted[0]]
        var cumulative = sorted[0].probability
        for candidate in sorted.dropFirst() {
            cumulative += candidate.probability
            guard cumulative <= topP else { break }
            kept.append(candidate)
        }
        let total = kept.reduce(0) { $0 + $1.probability }
        return kept.map { ($0.index, $0.probability / total) }
    }

    /// One index from `candidates`, drawn with `generator`.
    public func choose<G: RandomNumberGenerator>(logits: [Float], legal: [Int], using generator: inout G) -> Int? {
        let candidates = candidates(logits: logits, legal: legal)
        guard candidates.count > 1 else { return candidates.first?.index }
        let draw = Double.random(in: 0..<1, using: &generator)
        var cumulative = 0.0
        for candidate in candidates {
            cumulative += candidate.probability
            if draw < cumulative { return candidate.index }
        }
        return candidates.last!.index
    }

    /// Softmax of `logits / temperature` over the legal indices, most likely first (ties by
    /// index). Computed in Double.
    public static func softmax(logits: [Float], legal: [Int], temperature: Double = 1) -> [(index: Int, probability: Double)] {
        let scaled = legal.map { Double(logits[$0]) / temperature }
        let top = scaled.max() ?? 0
        let weights = scaled.map { exp($0 - top) }
        let total = weights.reduce(0, +)
        return zip(legal, weights)
            .map { (index: $0, probability: $1 / total) }
            .sorted { ($0.probability, -$0.index) > ($1.probability, -$1.index) }
    }
}

/// A small seedable generator (SplitMix64), so games against Maia can be replayed exactly.
public struct MaiaRandom: RandomNumberGenerator, Sendable, Hashable {
    private var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
