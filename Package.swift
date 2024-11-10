// swift-tools-version:5.9

import PackageDescription
import CompilerPluginSupport

var sources = [
    "src/llama.cpp",
    "src/llama-vocab.cpp",
    "src/llama-grammar.cpp",
    "src/llama-sampling.cpp",
    "src/unicode.cpp",
    "src/unicode-data.cpp",
    "ggml/src/ggml.c",
    "ggml/src/ggml-cpu.c",
    "ggml/src/ggml-alloc.c",
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-quants.c",
    "ggml/src/ggml-aarch64.c",
    "common/sampling.cpp",
    "common/common.cpp",
    "common/json-schema-to-grammar.cpp",
    "common/log.cpp",
    "common/console.cpp"
]

var resources: [Resource] = []
var linkerSettings: [LinkerSetting] = []
var cSettings: [CSetting] =  [
    .unsafeFlags(["-Wno-shorten-64-to-32", "-O3", "-DNDEBUG"]),
    .unsafeFlags(["-fno-objc-arc"]),
    // NOTE: NEW_LAPACK will required iOS version 16.4+
    // We should consider add this in the future when we drop support for iOS 14
    // (ref: ref: https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc)
    // .define("ACCELERATE_NEW_LAPACK"),
    // .define("ACCELERATE_LAPACK_ILP64")
]

#if canImport(Darwin)
sources.append("ggml/src/ggml-metal.m")
resources.append(.process("ggml/src/ggml-metal.metal"))
linkerSettings.append(.linkedFramework("Accelerate"))
cSettings.append(
    contentsOf: [
        .define("GGML_USE_ACCELERATE"),
        .define("GGML_USE_METAL")
    ]
)
#endif

#if os(Linux)
cSettings.append(.define("_GNU_SOURCE"))
#endif

let package = Package(
    name: "llama",
    platforms: [
        .macOS(.v13),
        .iOS(.v14),
        .watchOS(.v4),
        .tvOS(.v14)
    ],
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-syntax.git", branch: "main")
    ],
    targets: [
        .target(
            name: "llama",
            path: ".",
            exclude: [
                "build",
                "cmake",
                "examples",
                "scripts",
                "models",
                "tests",
                "CMakeLists.txt",
                "Makefile",
                "ggml/src/ggml-metal-embed.metal"
            ],
            sources: sources,
            resources: resources,
            publicHeadersPath: "spm-headers",
            cSettings: cSettings,
            linkerSettings: linkerSettings
        ),
        .target(name: "LlamaObjC",
                dependencies: ["llama"],
                path: "objc",
                sources: [
                    "CPUParams.mm",
                    "GGMLThreadpool.mm",
                    "GPTParams.mm",
                    "GPTSampler.mm",
                    "LlamaBatch.mm",
                    "LlamaObjC.mm",
                    "LlamaModel.mm",
                    "LlamaContext.mm",
                    "LlamaSession.mm",
                ],
                publicHeadersPath: "include",
                cSettings: cSettings,
                linkerSettings: linkerSettings),
        .macro(
            name: "JSONSchemaMacros",
            dependencies: [
                .product(name: "SwiftSyntax", package: "swift-syntax"),
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "swift/JSONSchemaMacros"
        ),
        .macro(
            name: "LlamaKitMacros",
            dependencies: [
                .product(name: "SwiftSyntax", package: "swift-syntax"),
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "swift/LlamaKitMacros"
        ),
        .target(
            name: "JSONSchema",
            dependencies: ["JSONSchemaMacros"],
            path: "swift/JSONSchema"
        ),
        .target(
            name: "LlamaKit",
            dependencies: ["JSONSchema", "LlamaObjC", "LlamaKitMacros"],
            path: "swift/LlamaKit"
        ),
        .testTarget(name: "LlamaKitTests",
                    dependencies: ["LlamaKit", "JSONSchema", "JSONSchemaMacros"],
                    path: "swift/test",
                    linkerSettings: [
                        .linkedFramework("XCTest"),
                        .linkedFramework("Testing")]),
        .executableTarget(name: "LlamaKitMain",
                          dependencies: ["LlamaKit"],
                          path: "swift/main",
                          resources: [.process("Llama-3.2-3B-Instruct-Q4_0.gguf")]),
    ],
    cxxLanguageStandard: .cxx17
)
