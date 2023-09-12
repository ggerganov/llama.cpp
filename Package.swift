// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "llama",
    platforms: [.macOS(.v11),
                .iOS(.v14),
                .watchOS(.v4),
                .tvOS(.v14)
    ],
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
                "ggml-metal.m"
            ],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32",
                              "-Wall",
                              "-Wextra",
                              "-Wpedantic",
                              "-Wshadow",
                              "-Wcast-qual",
                              "-Wstrict-prototypes",
                              "-Wpointer-arith",
                              "-Wdouble-promotion",
                              "-Wno-unused-function",
                              "-Wmissing-prototypes",
                              "-Werror=vla",
                              "-mavx",
                              "-mavx2",
                              "-mfma",
                              "-mf16c",
                              "-O3",
                              "-DNDEBUG",
                              "-Wno-format",
                              "-mf16c"]),
                .define("GGML_USE_K_QUANTS"),
                .define("GGML_USE_ACCELERATE"),
                .define("NDEBUG"),
                .define("_XOPEN_SOURCE", to: "600"),
                .define("_DARWIN_C_SOURCE"),
                .unsafeFlags(["-fno-objc-arc"]),
                .define("GGML_SWIFT"),
                .define("GGML_USE_METAL")
            ],
            cxxSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32",
                              "-Wall",
                              "-Wextra",
                              "-Wpedantic",
                              "-Wshadow",
                              "-Wcast-qual",
                              "-Wstrict-prototypes",
                              "-Wpointer-arith",
                              "-Wdouble-promotion",
                              "-Wno-unused-function",
                              "-Wmissing-prototypes",
                              "-Werror=vla",
                              "-mavx",
                              "-mavx2",
                              "-mfma",
                              "-mf16c",
                              "-O3",
                              "-DNDEBUG",
                              "-Wno-format",
                              "-mf16c"]),
                .define("GGML_USE_K_QUANTS"),
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags(["-fno-objc-arc"]),
                .define("GGML_SWIFT"),
                .define("GGML_USE_METAL"),
                .define("NDEBUG"),
                .define("_XOPEN_SOURCE", to: "600"),
                .define("_DARWIN_C_SOURCE")
                ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit")
            ]
        )
    ],
    cxxLanguageStandard: .cxx11
)
