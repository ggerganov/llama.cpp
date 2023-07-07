#!/bin/bash

pyinstaller --noconfirm --onefile --clean --console --collect-all customtkinter --icon "./niko.ico" \
--add-data "./klite.embd:." \
--add-data "./koboldcpp.so:." \
--add-data "./koboldcpp_openblas.so:." \
--add-data "./koboldcpp_failsafe.so:." \
--add-data "./koboldcpp_openblas_noavx2.so:." \
--add-data "./koboldcpp_clblast.so:." \
--add-data "./rwkv_vocab.embd:." \
--add-data "./rwkv_world_vocab.embd:." \
"./koboldcpp.py" -n "koboldcpp"
