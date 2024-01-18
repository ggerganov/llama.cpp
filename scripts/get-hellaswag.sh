#!/bin/bash

wget https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt

echo "Usage:"
echo ""
echo "  ./perplexity --hellaswag --hellaswag-tasks N -f hellaswag_val_full.txt -m modelfile.gguf"
echo ""

exit 0
