// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NCS2Kit",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "NCS2Kit", targets: ["NCS2Kit"]),
    ],
    targets: [
        .target(
            name: "NCS2Kit",
            path: "Sources/NCS2Kit"
        ),
        .testTarget(
            name: "NCS2KitTests",
            dependencies: ["NCS2Kit"],
            path: "Tests/NCS2KitTests"
        ),
    ]
)
