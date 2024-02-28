<!--

This source file is part of the Stanford Biodesign Digital Health Group open-source project.

SPDX-FileCopyrightText: 2022 Stanford University and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
  
-->

# Stanford BDHG llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This project is a Stanford BDHG-maintained fork of the well-regarded [llama.cpp](https://github.com/ggerganov/llama.cpp), tailored for deploying [LLaMA](https://arxiv.org/abs/2302.13971) models using C/C++. Our modifications package the library as an XCFramework for distribution as a binary compatible with multiple platforms. The inclusion of a `Package.swift` file facilitates the integration with the Swift Package Manager (SPM).

> [!NOTE]
> Should you have inquiries regarding the llama.cpp codebase this fork builds upon, please refer to the [upstream llama.cpp README](https://github.com/ggerganov/llama.cpp/blob/master/README.md) for comprehensive details and guidance.


## Setup

### Add Stanford BDHG llama.cpp as a Dependency

You need to add Stanford BDHG llama.cpp Swift package to
[your app in Xcode](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app#) or
[Swift package](https://developer.apple.com/documentation/xcode/creating-a-standalone-swift-package-with-xcode#Add-a-dependency-on-another-Swift-package).

> [!IMPORTANT]
> Important: In order to use the library, one needs to set build parameters in the consuming Xcode project or the consuming SPM package to enable the [Swift / C++ Interop](https://www.swift.org/documentation/cxx-interop/), introduced in Xcode 15 and Swift 5.9. Keep in mind that this is true for nested dependencies, one needs to set this configuration recursivly for the entire dependency tree towards the llama.cpp SPM package.
> 
> **For Xcode projects:**
> - Open your project settings in Xcode by selecting *PROJECT_NAME > TARGET_NAME > Build Settings*.
> - Within the *Build Settings*, search for the `C++ and Objective-C Interoperability` setting and set it to `C++ / Objective-C++`. This enables the project to use the C++ headers from llama.cpp.
>
> **For SPM packages:**
> - Open the `Package.swift` file of your SPM package
> - Within the package `target` that consumes the llama.cpp package, add the `interoperabilityMode(_:)` Swift build setting like that:
```swift
/// Adds the dependency to the Stanford BDHG llama.cpp SPM package
dependencies: [
    .package(url: "https://github.com/StanfordBDHG/llama.cpp", .upToNextMinor(from: "0.1.0"))
],
targets: [
  .target(
      name: "ExampleConsumingTarget",
      /// State the dependence of the target to llama.cpp
      dependencies: [
          .product(name: "llama", package: "llama.cpp")
      ],
      /// Important: Configure the `.interoperabilityMode(_:)` within the `swiftSettings`
      swiftSettings: [
          .interoperabilityMode(.Cxx)
      ]
  )
]
```

## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/StanfordBDHG/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/StanfordBDHG/.github/blob/main/CODE_OF_CONDUCT.md) first.
You can find a list of contributors in the [`CONTRIBUTORS.md`](https://github.com/StanfordBDHG/llama.cpp/blob/main/CONTRIBUTORS.md) file.

## License

This project is a fork of an existing project that is licensed under the MIT License, and all changes made in this fork continue to be under the MIT License. For more information about the license terms, see the [Licenses folder](https://github.com/StanfordBDHG/llama.cpp/blob/main/LICENSES).

## Our Research

For more information, check out our website at [biodesigndigitalhealth.stanford.edu](https://biodesigndigitalhealth.stanford.edu).

![Stanford Byers Center for Biodesign Logo](https://raw.githubusercontent.com/StanfordBDHG/.github/main/assets/biodesign-footer-light.png#gh-light-mode-only)
![Stanford Byers Center for Biodesign Logo](https://raw.githubusercontent.com/StanfordBDHG/.github/main/assets/biodesign-footer-dark.png#gh-dark-mode-only)