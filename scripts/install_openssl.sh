#!/bin/bash

# Check if OpenSSL 1.1 is installed
if ! (command -v openssl &> /dev/null && openssl version | grep -q "OpenSSL 1.1"); then

    # Check the architecture and install OpenSSL 1.1 if needed
    if [[ $(uname -m) == "arm64" ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # MacOS M1 installation
            if ! command -v brew &> /dev/null; then
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

                (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> "$HOME/.zprofile"

                eval "$(/opt/homebrew/bin/brew shellenv)"

                echo 'export PATH="/opt/homebrew/opt/openssl@1.1/bin:$PATH"' >> ~/.zshrc

            fi

            export PATH="/opt/homebrew/bin:$PATH"

            brew install openssl@1.1

        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Check for Debian-based system
            if grep -qi 'debian' /etc/os-release; then
                # Ubuntu ARM installation

                apt update && apt install curl -y

                curl -s https://packagecloud.io/install/repositories/wasmcloud/core/script.deb.sh | bash

                apt install wash

                curl -fLO http://ftp.us.debian.org/debian/pool/main/o/openssl/libssl1.1_1.1.1n-0+deb11u4_arm64.deb

                dpkg -i libssl1.1_1.1.1n-0+deb11u4_arm64.deb

            else
                echo "This script is designed for Debian-based systems only."
                exit 1
            fi

        else
            echo "Unsupported system type."
            exit 1
        fi

    else
        echo "This script is designed for arm64 systems only."
        exit 1
    fi

fi
