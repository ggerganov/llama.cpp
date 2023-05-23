/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>
#include "common.h"
#include "llama.h"

#include <grpcpp/grpcpp.h>

#include "absl/strings/str_format.h"

#ifdef BAZEL_BUILD
#include "examples/protos/message.grpc.pb.h"
#else
#include "message.grpc.pb.h"
#endif

// ABSL_FLAG(uint16_t, port, 50051, "Server port for the service");
// ABSL_FLAG(std::string, target, "localhost:50051", "Server address");

using grpc::CallbackServerContext;
using grpc::Server;
using grpc::ServerAsyncWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::ServerUnaryReactor;
using grpc::ServerWriteReactor;
using grpc::Status;
using robot::Job;
using robot::LlamaGoService;
using robot::Output;

struct server_params
{
  std::string hostname = "127.0.0.1";
  int32_t port = 8080;
};

void server_print_usage(int /*argc*/, char **argv, const gpt_params &params)
{
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
  fprintf(stderr, "  --memory_f32          use f32 instead of f16 for memory key+value\n");
  fprintf(stderr, "  --embedding           enable embedding mode\n");
  fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
  if (llama_mlock_supported())
  {
    fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
  }
  if (llama_mmap_supported())
  {
    fprintf(stderr, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
  }
  fprintf(stderr, "  -ngl N, --n-gpu-layers N\n");
  fprintf(stderr, "                        number of layers to store in VRAM\n");
  fprintf(stderr, "  -m FNAME, --model FNAME\n");
  fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
  fprintf(stderr, "  -host                 ip address to listen (default 127.0.0.1)\n");
  fprintf(stderr, "  -port PORT            port to listen (default 8080)\n");
  fprintf(stderr, "\n");
}

class LlamaServerContext
{
public:
  bool loaded;

  LlamaServerContext(gpt_params params_) : params(params_), threads(8)
  {
    ctx = llama_init_from_gpt_params(params);
    if (ctx == NULL)
    {
      loaded = false;
      fprintf(stderr, "%s: error: unable to load model\n", __func__);
    }
    else
    {
      loaded = true;
      // determine newline token
      llama_token_newline = ::llama_tokenize(ctx, "\n", false);
      last_n_tokens.resize(params.n_ctx);
      std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    }
  }

  std::vector<float> embedding(std::string content)
  {
    content.insert(0, 1, ' ');
    std::vector<llama_token> tokens = ::llama_tokenize(ctx, content, true);
    if (tokens.size() > 0)
    {
      fprintf(stderr, "---3---,%p,%d", ctx, threads);
      if (llama_eval(ctx, tokens.data(), tokens.size(), 0, 6))
      {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        std::vector<float> embeddings_;
        return embeddings_;
      }
    }
    const int n_embd = llama_n_embd(ctx);
    const auto embeddings = llama_get_embeddings(ctx);
    std::vector<float> embeddings_(embeddings, embeddings + n_embd);
    return embeddings_;
  }

private:
  gpt_params params;
  llama_context *ctx;
  int threads;

  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> llama_token_newline;
};

// Logic and data behind the server's behavior.
class LlamaServiceImpl final : public LlamaGoService::CallbackService
{

  class Reactor : public grpc::ServerWriteReactor<Output>
  {
  public:
    Reactor(CallbackServerContext *ctx, const Job *request)
        : ctx_(ctx), request_(request)
    {
      Output response;
      //  StartWrite(&response_);
      // StartWriteLast(&response_, WriteOptions());
      // ctx_->TryCancel();
    }
    void OnDone() override { delete this; }

  private:
    CallbackServerContext *const ctx_;
    const Job *const request_;
  };

public:
  LlamaServiceImpl(LlamaServerContext *llama_) : llama(llama_)
  {
    fprintf(stderr, "%s : new impl\n", __func__);
  }

  ServerWriteReactor<Output> *Answer(
      CallbackServerContext *context, const Job *request)
  {
    fprintf(stderr, "%s : get answer\n", __func__);
    return new Reactor(context, request);
  }

  ServerUnaryReactor *Embed(
      CallbackServerContext *context, const Job *request, Output *response)
  {
    fprintf(stderr, "%s : get embed %s\n", __func__, request->prompt().c_str());
    std::vector<float> embeded = llama->embedding(request->prompt());
    fprintf(stderr, "0");
    fprintf(stderr, "%p", embeded.begin());
    *response->mutable_embed() = {embeded.begin(), embeded.end()};
    fprintf(stderr, "1");
    response->set_id(request->id());
    fprintf(stderr, "2");
    ServerUnaryReactor *reactor = context->DefaultReactor();
    fprintf(stderr, "3");
    reactor->Finish(Status::OK);
    fprintf(stderr, "4");
    return reactor;
  }

private:
  LlamaServerContext *llama;
  int threads;
};

void RunServer(uint16_t port, LlamaServerContext *llama)
{
  std::string server_address = absl::StrFormat("0.0.0.0:%d", port);
  LlamaServiceImpl service(llama);

  grpc::EnableDefaultHealthCheckService(true);
  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

bool server_params_parse(int argc, char **argv, server_params &sparams, gpt_params &params)
{
  gpt_params default_params;
  std::string arg;
  bool invalid_param = false;

  for (int i = 1; i < argc; i++)
  {
    arg = argv[i];
    if (arg == "--port")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      sparams.port = std::stoi(argv[i]);
    }
    else if (arg == "--host")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      sparams.hostname = argv[i];
    }
    else if (arg == "-s" || arg == "--seed")
    {
#if defined(GGML_USE_CUBLAS)
      fprintf(stderr, "WARNING: when using cuBLAS generation results are NOT guaranteed to be reproducible.\n");
#endif
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.seed = std::stoi(argv[i]);
    }
    else if (arg == "-m" || arg == "--model")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.model = argv[i];
    }
    else if (arg == "--embedding")
    {
      params.embedding = true;
    }
    else if (arg == "-h" || arg == "--help")
    {
      server_print_usage(argc, argv, default_params);
      exit(0);
    }
    else if (arg == "-c" || arg == "--ctx_size")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.n_ctx = std::stoi(argv[i]);
    }
    else if (arg == "--memory_f32")
    {
      params.memory_f16 = false;
    }
    else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.n_gpu_layers = std::stoi(argv[i]);
    }
    else
    {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      server_print_usage(argc, argv, default_params);
      exit(1);
    }
  }

  if (invalid_param)
  {
    fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
    server_print_usage(argc, argv, default_params);
    exit(1);
  }
  return true;
}

int main(int argc, char **argv)
{

  gpt_params params;
  server_params sparams;

  llama_init_backend();

  params.model = "ggml-model.bin";
  params.n_ctx = 512;

  sparams.port = 8080;

  if (gpt_params_parse(argc, argv, params) == false)
  {
    return 1;
  }

  params.embedding = true;

  if (params.seed <= 0)
  {
    params.seed = time(NULL);
  }

  fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

  LlamaServerContext llama(params);

  // load the model
  if (!llama.loaded)
  {
    return 1;
  }

  RunServer(sparams.port, &llama);
  return 0;
}
