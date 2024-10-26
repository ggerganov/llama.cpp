// swift-tools-version: 5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "jarvis-batched-swift",
    platforms: [.macOS(.v12)],
    dependencies: [
        .package(name: "jarvis", path: "../../"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "jarvis-batched-swift",
            dependencies: ["jarvis"],
            path: "Sources",
            linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("AppKit")]
        ),
    ]
)
