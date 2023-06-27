#!/bin/bash
# Download and update deps for binary

# get the directory of this script file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PUBLIC=$DIR/public

curl https://npm.reversehttp.com/@preact/signals-core,@preact/signals,htm/preact,preact,preact/hooks,@microsoft/fetch-event-source > $PUBLIC/index.js
echo >> $PUBLIC/index.js # add newline

echo "// Generated file, run deps.sh to update. Do not edit directly
R\"htmlraw($(cat $PUBLIC/index.html))htmlraw\"
" > $DIR/index.html.cpp

echo "// Generated file, run deps.sh to update. Do not edit directly
R\"jsraw($(cat $PUBLIC/index.js))jsraw\"
" > $DIR/index.js.cpp
