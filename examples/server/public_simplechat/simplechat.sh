
PORT=$1

if [ "$PORT" == "" ]; then
	PORT=9000
fi

echo "Open http://127.0.0.1:$PORT/simplechat.html in your local browser"
python3 -m http.server $PORT
