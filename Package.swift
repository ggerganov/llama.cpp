// swift-tools-version:5.3

import PackageDescription

let unsafeFlags = ["-Wno-shorten-64-to-32", "-I/opt/homebrew/opt/clblast/include"]
let defines = ["GGML_USE_ACCELERATE", "GGML_USE_CLBLAST"]

let package = Package(
    name: "llama",
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .target(
            name: "llama",
            path: ".",
            sources: ["ggml.c", "llama.cpp", "ggml-opencl.c"],
            publicHeadersPath: "spm-headers",
            cSettings: [.unsafeFlags(unsafeFlags)] + defines.map { .define($0) },
            cxxSettings: [.unsafeFlags(unsafeFlags)] + defines.map { .define($0) },
            linkerSettings: [.linkedFramework("Accelerate"), .linkedFramework("OpenCL")]
        ),
    ],
    cxxLanguageStandard: .cxx11
)
