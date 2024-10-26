.PHONY: build

build:
	xcodebuild -scheme jarvis-batched-swift -destination "generic/platform=macOS" -derivedDataPath build
	rm -f ./jarvis-batched-swift
	ln -s ./build/Build/Products/Debug/jarvis-batched-swift ./jarvis-batched-swift
