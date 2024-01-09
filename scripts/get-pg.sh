#!/bin/bash

function usage {
    echo "usage: <n>$0"
    exit 1
}

function has_cmd {
    if ! [ -x "$(command -v $1)" ]; then
        echo "error: $1 is not available" >&2
        exit 1
    fi
}

# check for: curl, html2text, tail, sed, fmt
has_cmd curl
has_cmd html2text
has_cmd tail
has_cmd sed

if [ $# -ne 1 ]; then
    usage
fi

n=$1

# get urls
urls="$(curl http://www.aaronsw.com/2002/feeds/pgessays.rss | grep html | sed -e "s/.*http/http/" | sed -e "s/html.*/html/" | head -n $n)"

printf "urls:\n%s\n" "$urls"

if [ -f pg.txt ]; then
    rm pg.txt
fi

for url in $urls; do
    echo "processing $url"

    curl -L $url | html2text | tail -n +4 | sed -E "s/^[[:space:]]+//g" | fmt -w 80 >> pg.txt

    # don't flood the server
    sleep 1
done

echo "done. data in pg.txt"

exit 0
