.PHONY: build

build:
	xcodebuild -scheme batched_swift -destination "generic/platform=macOS" -derivedDataPath build
	rm -f ./batched_swift
	ln -s ./build/Build/Products/Debug/batched_swift ./batched_swift
