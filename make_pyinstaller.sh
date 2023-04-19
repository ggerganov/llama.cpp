#!/bin/bash

pyinstaller --noconfirm --onefile --clean --console --icon "./niko.ico" \
--add-data "./klite.embd:." \
--add-data "./koboldcpp.dll:." \
--add-data "./ggml_openblas.o:." \
--add-data "./ggml_noavx2.o:." \
--add-data "./ggml_openblas_noavx2.o:." \
--add-data "./libopenblas.dll:." \
--add-data "./ggml_clblast.o:." \
--add-data "./clblast.dll:." \
--add-data "./rwkv_vocab.embd:." \
"./koboldcpp.py" -n "koboldcpp"
