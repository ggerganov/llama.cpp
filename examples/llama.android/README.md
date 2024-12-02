# Nexa

**Nexa** is a Kotlin wrapper for the [llama.cpp](https://github.com/ggerganov/llama.cpp.git) library. offering a convenient Kotlin API for Android developers. It allows seamless integration of llama.cpp models into Android applications.
**NOTE:** Currently, Nexa supports Vision-Language Model (VLM) inference capabilities.

## Installation

To add Nexa to your Android project, follow these steps:

- Create a libs folder in your project’s root directory.
- Copy the .aar file into the libs folder.
- Add dependency to your build.gradle file:

```
implementation files("libs/com.nexa.aar")
```

## Usage
### 1. Initialize NexaSwift with model path and projector path

Create a configuration and initialize NexaSwift with the path to your model file:

```kotlin
nexaVlmInference = NexaVlmInference(pathToModel,
    mmprojectorPath, imagePath,
    maxNewTokens = 128,
    stopWords = listOf("</s>"))
nexaVlmInference.loadModel()
```

### 2. Completion API

#### Streaming Mode

```swift
nexaVlmInference.createCompletionStream(prompt, imagePath)
    ?.catch {
        print(it.message)
    }
    ?.collect { print(it) }
```

### 3. release all resources
```kotlin
nexaVlmInference.dispose()
```

## Quick Start

Open the [android test project](./app-java) folder in Android Studio and run the project.

## Download Models

You can download models from the [Nexa AI ModelHub](https://nexa.ai/models).

## How to estimate power usage

- ```adb shell dumpsys batterystats --reset```
- ```adb shell dumpsys batterystats > batterystats.txt```