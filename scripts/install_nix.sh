#!/bin/bash

# Check if Nix is installed
if ! command -v nix --help >/dev/null 2>&1; then
  # Delete problematic backup files for bash and zsh if they exist
  if [ -f "/etc/bash.bashrc.backup-before-nix" ]; then
    sudo rm -f /etc/bash.bashrc.backup-before-nix
  fi

  if [ -f "/etc/bashrc.backup-before-nix" ]; then
    sudo rm -f /etc/bashrc.backup-before-nix
  fi

  if [ -f "/etc/zshrc.backup-before-nix" ]; then
    sudo rm -f /etc/zshrc.backup-before-nix
  fi

  # Determine the platform (Linux or macOS)
  case `uname` in
    Linux*)
      echo "Error: Nix package manager is not installed. Installing Nix for Linux..."
      sh <(curl -L https://nixos.org/nix/install) --daemon
      ;;
    Darwin*)
      echo "Error: Nix package manager is not installed. Installing Nix for macOS..."
      sh <(curl -L https://nixos.org/nix/install) --darwin-use-unencrypted-nix-store-volume
      ;;
    *)
      echo "Unsupported platform for Nix installation"
      exit 1;
      ;;
  esac

else
  echo "Nix package manager is already installed."
fi
