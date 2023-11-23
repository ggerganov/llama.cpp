// swift-tools-version:5.3

import PackageDescription

#if arch(arm) || arch(arm64)
let platforms: [SupportedPlatform]? = [
    .macOS(.v11),
    .iOS(.v14),
    .watchOS(.v4),
    .tvOS(.v14)
]
let exclude: [String] = []
let resources: [Resource] = [
    .process("ggml-metal.metal")
]
let additionalSources: [String] = ["ggml-metal.m"]
let additionalSettings: [CSetting] = [
    .unsafeFlags(["-fno-objc-arc"]),
    .define("GGML_USE_METAL")
]
#else
let platforms: [SupportedPlatform]? = nil
let exclude: [String] = ["ggml-metal.metal"]
let resources: [Resource] = []
let additionalSources: [String] = []
let additionalSettings: [CSetting] = []
#endif

let package = Package(
    name: "llama",
    platforms: platforms,
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .target(
            name: "llama",
            path: ".",
            exclude: exclude,
            sources: [
                "ggml.c",
                "llama.cpp",
                "ggml-alloc.c",
                "k_quants.c",
            ] + additionalSources,
            resources: resources,
            publicHeadersPath: "spm-headers",
            cSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
                .define("GGML_USE_K_QUANTS"),
                .define("GGML_USE_ACCELERATE")
                // NOTE: NEW_LAPACK will required iOS version 16.4+
                // We should consider add this in the future when we drop support for iOS 14
                // (ref: ref: https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc)
                // .define("ACCELERATE_NEW_LAPACK"),
                // .define("ACCELERATE_LAPACK_ILP64")
            ] + additionalSettings,
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        )
    ],
    cxxLanguageStandard: .cxx11
)
