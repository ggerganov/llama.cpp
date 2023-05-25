#!/bin/bash

# Detect the current shell
current_shell=$(basename "$SHELL")

# Run the appropriate command based on the detected shell
case $current_shell in
    bash)
        source ~/.bashrc || source ~/.bash_profile
        ;;
    zsh)
        if [ -f ~/.zshrc ]; then
            source ~/.zshrc
        else
            touch ~/.zshrc
            source ~/.zshrc
        fi
        ;;
    *)
        exit 1
        ;;
esac
