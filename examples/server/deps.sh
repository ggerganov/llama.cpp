#!/bin/bash
# Download and update deps for binary

# get the directory of this script file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PUBLIC=$DIR/public

echo "download js bundle files"
curl https://npm.reversehttp.com/@preact/signals-core,@preact/signals,htm/preact,preact,preact/hooks > $PUBLIC/index.js
echo >> $PUBLIC/index.js # add newline


URLS=(
    'https://cdn.jsdelivr.net/npm/zero-md@2.5.3/dist/zero-md.min.js'
    'https://cdn.jsdelivr.net/gh/markedjs/marked@4/marked.min.js'
    'https://cdn.jsdelivr.net/npm/github-markdown-css@5.5.0/github-markdown.min.css'
    'https://registry.npmjs.org/prismjs/-/prismjs-1.29.0.tgz'
    'https://cdn.jsdelivr.net/gh/katorlys/prism-theme-github/themes/prism-theme-github-light.css'
    'https://cdn.jsdelivr.net/gh/katorlys/prism-theme-github/themes/prism-theme-github-dark.css'
)

echo "download js bundle files"
for url in "${URLS[@]}"; do
  filename=$PUBLIC/assets/$(basename "$url")
  curl -o $filename --create-dirs $url
  echo >> $filename # add newline
done


tar -xvf $PUBLIC/assets/prismjs-1.29.0.tgz -C $PUBLIC/assets/
mv $PUBLIC/assets/package $PUBLIC/assets/prismjs
rm $PUBLIC/assets/prismjs-1.29.0.tgz


FILES=$(ls $PUBLIC)

cd $PUBLIC

for FILE in $FILES; do
  echo "generate $FILE.hpp"

  # use simple flag for old version of xxd
  xxd -i $FILE > $DIR/$FILE.hpp
done
