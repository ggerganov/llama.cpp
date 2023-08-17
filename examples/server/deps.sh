#!/bin/bash
# Download and update deps for binary

# get the directory of this script file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PUBLIC=$DIR/public

if [ "$1" != "--no-download" ]; then
  echo "download js bundle files"
  curl https://npm.reversehttp.com/@preact/signals-core,@preact/signals,htm/preact,preact,preact/hooks > $PUBLIC/index.js
  echo >> $PUBLIC/index.js # add newline
fi

FILES=$(ls $PUBLIC)
UNAME_S=$(uname -s)

for FILE in $FILES; do
  func=$(echo $FILE | tr '.' '_')
  echo "generate $FILE.hpp ($func)"

  if [ "$UNAME_S" == "Darwin" ]; then
    xxd -n $func -i $PUBLIC/$FILE > $DIR/$FILE.hpp
  elif [ "$UNAME_S" == "Linux" ]; then
    xxd -i $PUBLIC/$FILE > $DIR/$FILE.hpp
    replace_prefix="$(echo $PUBLIC | tr '/' '_')_"
    sed -i "s/$replace_prefix//g" $DIR/$FILE.hpp
  fi
done
