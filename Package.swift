// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "llama",
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .target(
            name: "llama",
            path: ".",
            exclude: ["ggml-metal.metal"],
            sources: [
                "ggml.c",
                "llama.cpp",
                "ggml-alloc.c",
                "k_quants.c",
                "common/common.cpp",
                "common/console.cpp",
                "common/grammar-parser.cpp",
            ],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .headerSearchPath("common"),
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
                .define("GGML_USE_K_QUANTS"),
                .define("GGML_USE_ACCELERATE"),
                .define("LLAMA_USE_SWIFT")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),
    ],
    cxxLanguageStandard: .cxx11
)
