// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "llama",
    platforms: [.macOS(.v11)],
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .target(
            name: "llama",
            path: ".",
            sources: [
                "ggml.c",
                "llama.cpp",
                "ggml-alloc.c",
                "k_quants.c",
                "ggml-metal.m",
            ],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
                .unsafeFlags(["-fno-objc-arc"]),
                .define("GGML_SWIFT"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_K_QUANTS"),
                .define("GGML_USE_ACCELERATE")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),
    ],
    cxxLanguageStandard: .cxx11
)
