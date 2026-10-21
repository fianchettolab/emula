// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Emula",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Emula", targets: ["Emula"]),
    ],
    dependencies: [
        // The chess rules: the shakmaty bridge, checked out next to this one.
        .package(path: "../scacchi"),
    ],
    targets: [
        .target(name: "Emula", dependencies: [.product(name: "Scacchi", package: "scacchi")]),
        .testTarget(name: "EmulaTests", dependencies: ["Emula"]),
    ]
)
