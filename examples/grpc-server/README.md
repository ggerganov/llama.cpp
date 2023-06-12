# llama grpc server

service as a grpc server to completion and embedding (when `--embedding` argument is given) based on examples/server.

## running service

run grpc-server command using argument like main program of llama.cpp with the following change:

* add `--host` argument to set the listening host
* add `--port` argument to set the listening port

### behaving differences with examples/server

* grpc-server will always break when <eos> is the predicted token.