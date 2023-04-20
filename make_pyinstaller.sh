#!/bin/bash

pyinstaller --noconfirm --onefile --clean --console --icon "./niko.ico" \
--add-data "./klite.embd:." \
--add-data "./koboldcpp.so:." \
--add-data "./koboldcpp_openblas.so:." \
--add-data "./koboldcpp_noavx2.so:." \
--add-data "./koboldcpp_openblas_noavx2.so:." \
--add-data "./koboldcpp_clblast.so:." \
--add-data "./rwkv_vocab.embd:." \
"./koboldcpp.py" -n "koboldcpp"
