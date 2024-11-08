#!/bin/bash
# Download and update deps for binary

# get the directory of this script file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PUBLIC=$DIR/public

echo "download js bundle files"

# Note for contributors: Always pin to a specific version "maj.min.patch" to avoid breaking the CI

curl -L https://cdn.tailwindcss.com/3.4.14 > $PUBLIC/deps_tailwindcss.js
echo >> $PUBLIC/deps_tailwindcss.js # add newline

curl -L https://cdnjs.cloudflare.com/ajax/libs/daisyui/4.12.14/styled.min.css > $PUBLIC/deps_daisyui.min.css
curl -L https://cdnjs.cloudflare.com/ajax/libs/daisyui/4.12.14/themes.min.css >> $PUBLIC/deps_daisyui.min.css
echo >> $PUBLIC/deps_daisyui.min.css # add newline

curl -L https://unpkg.com/vue@3.5.12/dist/vue.esm-browser.js > $PUBLIC/deps_vue.esm-browser.js
echo >> $PUBLIC/deps_vue.esm-browser.js # add newline

curl -L https://cdnjs.cloudflare.com/ajax/libs/markdown-it/13.0.2/markdown-it.js > $PUBLIC/deps_markdown-it.js
echo >> $PUBLIC/deps_markdown-it.js # add newline

ls -lah $PUBLIC
