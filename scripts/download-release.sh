#!/bin/bash

# Downloads a release from the llama.cpp repository.
#
# Example usage:
#   curl -s https://raw.githubusercontent.com/ggerganov/llama.cpp/master/download-release.sh | bash -s -- /path/to/dest --tag=latest
#   chmod +x /path/to/dest/llama-server
#   /path/to/dest/llama-server --help
#
# list of release zips available as of release b3190 (2024-06-20):
# cudart-llama-bin-win-cu11.7.1-x64.zip
# cudart-llama-bin-win-cu12.2.0-x64.zip
# llama-b3190-bin-macos-arm64.zip
# llama-b3190-bin-macos-x64.zip
# llama-b3190-bin-ubuntu-x64.zip
# llama-b3190-bin-win-avx-x64.zip
# llama-b3190-bin-win-avx2-x64.zip
# llama-b3190-bin-win-avx512-x64.zip
# llama-b3190-bin-win-cuda-cu11.7.1-x64.zip
# llama-b3190-bin-win-cuda-cu12.2.0-x64.zip
# llama-b3190-bin-win-kompute-x64.zip
# llama-b3190-bin-win-llvm-arm64.zip
# llama-b3190-bin-win-msvc-arm64.zip
# llama-b3190-bin-win-noavx-x64.zip
# llama-b3190-bin-win-openblas-x64.zip
# llama-b3190-bin-win-rpc-x64.zip
# llama-b3190-bin-win-sycl-x64.zip
# llama-b3190-bin-win-vulkan-x64.zip

parse_args() {
    if [ "$#" -lt 1 ]; then
        print_help
        exit 1
    fi
    DEST_DIR="$1"
    shift
    for i in "$@"; do
        case $i in
        --help)
            print_help
            exit 0
            ;;
        --dest=*)
            DEST_DIR="${i#*=}"
            shift
            ;;
        --tag=*)
            TAG="${i#*=}"
            shift
            ;;
        --filename=*)
            FILENAME="${i#*=}"
            shift
            ;;
        --os=*)
            OS="${i#*=}"
            shift
            ;;
        --arch=*)
            ARCH="${i#*=}"
            shift
            ;;
        --backend=*)
            BACKEND="${i#*=}"
            shift
            ;;
        *)
            print_help
            exit 1
            ;;
        esac
    done
    # if not set, default to latest
    TAG=${TAG:-latest}
    BACKEND=${BACKEND:-cuda-cu12.2.0}
    get_tag
    get_os
    get_arch
    get_filename
}

print_help() {
    echo "Usage: download-llama-cpp.sh --help"
    echo "       download-llama-cpp.sh <dest> --tag=<tag> --filename=<filename> --os=<os> --arch=<arch> --backend=<backend>"
    echo "Options:"
    echo "  <dest>                   Destination directory to download the files to."
    echo "                           e.g. the llama-server executable will be at <dest>/llama-server"
    echo "  --tag=<tag>              Release tag to download. Default: latest"
    echo "  --filename=<filename>    Filename to download. If supplied, will"
    echo "                           override the os/arch/backend options."
    echo "                           If not supplied,"
    echo "                           will be constructed from the os/arch/backend options."
    echo "  --os=<os>                OS to download for. Default: auto-detected"
    echo "  --arch=<arch>            Architecture to download for. Default: auto-detected"
    echo "  --backend=<backend>      Backend to download for. Only relevant for Windows. Default: cuda-cu12.2.0"
}

get_tag() {
    # if not latest, return
    if [ "$TAG" != "latest" ]; then
        return
    fi
    # extract the tag from the latest release:
    # "tag_name": "v1.0.0",
    # Trying to avoid needing jq or a real json parser
    TAG=$(
        curl -s "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest" |
            # there can be a variable amount of whitespace between the key and value
            sed -n 's/.*"tag_name":[ \n]*"\([^"]*\)".*/\1/p'
    )
    echo "Latest release tag found to be: $TAG"
}

get_os() {
    # if already set, return
    if [ -n "$OS" ]; then
        return
    fi
    OS=$(uname -s)

    case "$OS" in
    Linux)
        OS="ubuntu"
        ;;
    Darwin)
        OS="macos"
        ;;
    CYGWIN* | MINGW32* | MSYS* | MINGW*)
        OS="win"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
    esac
}

get_arch() {
    # if already set, return
    if [ -n "$ARCH" ]; then
        return
    fi
    ARCH=$(uname -m)

    case "$ARCH" in
    x86_64)
        ARCH="x64"
        ;;
    arm64)
        ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
    esac
}

get_filename() {
    if [ -n "$FILENAME" ]; then
        return
    fi
    if [[ "$OS" == "ubuntu" ]]; then
        FILENAME="llama-$TAG-bin-ubuntu-$ARCH.zip"
    elif [[ "$OS" == "macos" ]]; then
        FILENAME="llama-$TAG-bin-macos-$ARCH.zip"
    elif [[ "$OS" == "win" ]]; then
        FILENAME="llama-$TAG-bin-win-$BACKEND-$ARCH.zip"
    fi
}

ensure_downloaded() {
    # only download if the file does not exist
    if [ -f "$ZIP_PATH" ]; then
        echo "File $ZIP_PATH already exists, skipping download"
        return
    fi
    local base_url="https://github.com/ggerganov/llama.cpp/releases/download"
    URL="$base_url/$TAG/$FILENAME"
    echo "Downloading $URL to $ZIP_PATH"
    curl -L -o "$ZIP_PATH" "$URL"
}

main() {
    parse_args "$@"

    mkdir -p "$DEST_DIR"
    ZIP_PATH="$DEST_DIR/$FILENAME"
    ensure_downloaded

    echo "Unzipping $ZIP_PATH to $DEST_DIR"
    unzip "$ZIP_PATH" -d "$DEST_DIR"
    # All of the contents are in a build/bin subdirectory,
    # move all of the contents up to the DEST_DIR
    mv "$DEST_DIR/build/bin"/* "$DEST_DIR"
    rmdir "$DEST_DIR/build/bin" "$DEST_DIR/build"

    echo "Done!"
}

main "$@"
