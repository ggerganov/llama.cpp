// swift-tools-version:5.5

import PackageDescription

var sources = [
   "src/llama.cpp",
   "src/llama-vocab.cpp",
   "src/llama-grammar.cpp",
   "src/llama-sampling.cpp",
   "src/unicode.cpp",
   "src/unicode-data.cpp",
   "ggml/src/ggml.c",
   "ggml/src/ggml-aarch64.c",
   "ggml/src/ggml-alloc.c",
   "ggml/src/ggml-backend.cpp",
   "ggml/src/ggml-backend-reg.cpp",
   "ggml/src/ggml-cpu/ggml-cpu.c",
   "ggml/src/ggml-cpu/ggml-cpu.cpp",
   "ggml/src/ggml-cpu/ggml-cpu-aarch64.c",
   "ggml/src/ggml-cpu/ggml-cpu-quants.c",
   "ggml/src/ggml-threading.cpp",
   "ggml/src/ggml-quants.c",
]

var omniVlmSources = [
   "common/log.h",
   "common/log.cpp",
   "common/arg.h",
   "common/arg.cpp",
   "common/common.cpp", 
   "common/common.h", 
   "common/json.hpp", 
   "common/json-schema-to-grammar.cpp", 
   "common/json-schema-to-grammar.h", 
   "src/llama-grammar.h",
   "common/grammar-parser.cpp",
   "common/grammar-parser.h",
   "common/sampling.cpp",
   "common/sampling.h", 
   "examples/omni-vlm/build-info.cpp",
   "examples/omni-vlm/clip.cpp",
   "examples/omni-vlm/clip.h",
   "examples/omni-vlm/omni-vlm-wrapper.cpp",
   "examples/omni-vlm/omni-vlm-wrapper.h",
   "examples/omni-vlm/omni-vlm.h",
   "examples/omni-vlm/omni-vlm.cpp",
   "common/base64.cpp",
   "ggml/include/ggml.h",
   "ggml/include/ggml-alloc.h",
   "ggml/include/ggml-backend.h",
   "ggml/src/ggml-common.h",
]

var testSources = [
   "tests/LlavaTests/LlavaTests.swift"
]

var resources: [Resource] = []
var linkerSettings: [LinkerSetting] = []
var cSettings: [CSetting] = [
   .unsafeFlags(["-Wno-shorten-64-to-32", "-O3", "-DNDEBUG"]),
   .unsafeFlags(["-fno-objc-arc"]),
   .headerSearchPath("."),
   .headerSearchPath("ggml/src"),
   .headerSearchPath("common"),
   .unsafeFlags(["-framework", "Foundation"]),  
   .unsafeFlags(["-framework", "Accelerate"]), 
]

#if os(Linux)
   cSettings.append(.define("_GNU_SOURCE"))
#endif

let baseSettings = cSettings + [
   .headerSearchPath("."),  
   .headerSearchPath("src"),  
   .headerSearchPath("common"),
   .headerSearchPath("examples/omni-vlm"),
   .headerSearchPath("ggml/include"),   
]

let llamaTarget = Target.target(
   name: "llama",
   dependencies: [],
   path: ".",
   exclude: [
       "build", "cmake", "examples", "scripts", "models", 
       "tests", "CMakeLists.txt", "Makefile",
   ],
   sources: sources,
   resources: resources,
   publicHeadersPath: "spm-headers",
   cSettings: cSettings,
   linkerSettings: linkerSettings
)

let omnivlmTarget = Target.target(
   name: "omnivlm",
   dependencies: ["llama"],
   path: ".",  
   sources: omniVlmSources,
   resources: resources,
   publicHeadersPath: "spm/omnivlm",
   cSettings: baseSettings + [
    .headerSearchPath("ggml/src"), 
   ],
   cxxSettings: [.unsafeFlags(["-std=c++14"])],
   linkerSettings: linkerSettings
)

let testTarget = Target.testTarget(
   name: "LlavaTests",
   dependencies: ["omnivlm"],
   path: ".",
   sources: testSources,
   resources: resources,
   cSettings: baseSettings + [
    .headerSearchPath("ggml/src"), 
   ],
   linkerSettings: linkerSettings
)

let supportedPlatforms: [SupportedPlatform] = [
    .macOS(.v12),
    .iOS(.v14),
    .watchOS(.v4),
    .tvOS(.v14)
]

let products = [
    Product.library(name: "llama", targets: ["llama"]),
    Product.library(name: "omnivlm", targets: ["omnivlm"])
]

let package = Package(
   name: "llama",
   platforms: supportedPlatforms,
   products: products,
   targets: [
       llamaTarget,
       omnivlmTarget,
       testTarget
   ],
   cxxLanguageStandard: .cxx14
)
