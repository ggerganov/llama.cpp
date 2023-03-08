#https://github.com/google/sentencepiece.git
#9ffb33a14c97c512103be0ee74740099660b39aa

curl -LO https://github.com/google/sentencepiece/releases/download/v0.1.97/sentencepiece-0.1.97.tar.gz
tar xzvf sentencepiece-0.1.97.tar.gz
cd sentencepiece-0.1.97/src
mkdir build
cd build
cmake ..
make sentencepiece-static -j $(nproc)
cd ../..

