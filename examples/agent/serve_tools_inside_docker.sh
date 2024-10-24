#!/bin/bash
#
# Serves tools inside a docker container.
#
# All outgoing HTTP *and* HTTPS traffic will be logged to `examples/agent/squid/logs/access.log`.
# Direct traffic to the host machine will be ~blocked, but clever AIs may find a way around it:
# make sure to have proper firewall rules in place.
#
# Take a look at `examples/agent/squid/conf/squid.conf` if you want tools to access your local llama-server(s).
#
# Usage:
#   examples/agent/serve_tools_inside_docker.sh
#
set -euo pipefail

cd examples/agent

mkdir -p squid/{cache,logs,ssl_cert,ssl_db}
rm -f squid/logs/{access,cache}.log

# Generate a self-signed certificate for the outgoing proxy.
# Tools can only reach out to HTTPS endpoints through that proxy, which they are told to trust blindly.
openssl req -new -newkey rsa:4096 -days 3650 -nodes -x509 \
    -keyout squid/ssl_cert/squidCA.pem \
    -out squid/ssl_cert/squidCA.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=Org Unit/CN=outgoing_proxy"

openssl x509 -outform PEM -in squid/ssl_cert/squidCA.pem -out squid/ssl_cert/squidCA.crt

docker compose up --build "$@"
