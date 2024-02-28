// swift-tools-version:5.9

//
// This source file is part of the TemplatePackage open source project
//
// SPDX-FileCopyrightText: 2022 Stanford University and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import PackageDescription


let package = Package(
    name: "llama",
    platforms: [
        .iOS(.v14)
    ],
    products: [
        .library(
            name: "llama",
            targets: [
                "llama"
            ]
        )
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "./llama.xcframework"
        )
    ]
)
