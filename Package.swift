// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Emula",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Emula", targets: ["Emula"]),
    ],
    dependencies: [
        // The chess rules: the shakmaty bridge.
        .package(url: "https://github.com/fianchettolab/scacchi", from: "0.1.0"),
    ],
    targets: [
        .target(name: "Emula", dependencies: [.product(name: "Scacchi", package: "scacchi")]),
        .testTarget(name: "EmulaTests", dependencies: ["Emula"]),
    ]
)
