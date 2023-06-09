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

using grpc::CallbackServerContext;
using grpc::Server;
using grpc::ServerAsyncWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::ServerUnaryReactor;
using grpc::ServerWriteReactor;
using grpc::Status;
using llama::Job;
using llama::LlamaGoService;
using llama::Output;

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
  bool has_next_token{false};
  int32_t num_tokens_predicted{0};
  int32_t n_past{0};
  int32_t n_consumed{0};
  int32_t n_session_consumed{0};
  int32_t n_remain{0};

  std::vector<llama_token> embd;
  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> processed_tokens;
  std::vector<llama_token> llama_token_newline;
  std::vector<llama_token> embd_inp;
  std::vector<std::vector<llama_token>> no_show_words;
  std::vector<llama_token> tokens_predicted;

  LlamaServerContext(gpt_params params_) : params(params_), threads(8)
  {
    bool has_embedding = params.embedding;
    if (params.embedding)
    {
      ctx_for_embedding = llama_init_from_gpt_params(params);
    }
    params.embedding = false;
    fprintf(stderr, "%s: loading model\n", __func__);
    ctx = llama_init_from_gpt_params(params);
    if (ctx == NULL || (has_embedding && ctx_for_embedding == NULL))
    {
      loaded = false;
      fprintf(stderr, "%s: error: unable to load model\n", __func__);
    }
    else
    {
      fprintf(stderr, "%s: model loaded\n", __func__);
      loaded = true;
      last_n_tokens.resize(params.n_ctx);
      std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    }
  }

  std::vector<float> embedding(std::string content)
  {
    content.insert(0, 1, ' ');
    std::vector<llama_token> tokens = ::llama_tokenize(ctx_for_embedding, content, true);
    if (tokens.size() > 0)
    {
      if (llama_eval(ctx_for_embedding, tokens.data(), tokens.size(), 0, 6))
      {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        std::vector<float> embeddings_;
        return embeddings_;
      }
    }
    const int n_embd = llama_n_embd(ctx_for_embedding);
    const auto embeddings = llama_get_embeddings(ctx_for_embedding);
    std::vector<float> embeddings_(embeddings, embeddings + n_embd);
    return embeddings_;
  }

  void rewind()
  {
    // as_loop = false;
    params.antiprompt.clear();
    no_show_words.clear();
    num_tokens_predicted = 0;
    // embd.clear();
    // n_remain = 0;
    // generated_text = "";
  }

  std::string doCompletion()
  {
    llama_token token = nextToken();
    if (token == -1)
    {
      return "";
    }
    tokens_predicted.clear();
    tokens_predicted.push_back(token);

    // Avoid add the no show words to the response
    for (std::vector<llama_token> word_tokens : no_show_words)
    {
      int match_token = 1;
      if (tokens_predicted.front() == word_tokens.front())
      {
        bool execute_matching = true;
        if (tokens_predicted.size() > 1)
        { // if previus tokens had been tested
          for (int i = 1; i < word_tokens.size(); i++)
          {
            if (i >= tokens_predicted.size())
            {
              match_token = i;
              break;
            }
            if (tokens_predicted[i] == word_tokens[i])
            {
              continue;
            }
            else
            {
              execute_matching = false;
              break;
            }
          }
        }
        while (execute_matching)
        {
          if (match_token == word_tokens.size())
          {
            return "";
          }
          token = nextToken();
          tokens_predicted.push_back(token);
          if (token == word_tokens[match_token])
          { // the token follow the sequence
            match_token++;
          }
          else if (match_token < word_tokens.size())
          { // no complete all word sequence
            break;
          }
        }
      }
    }

    std::string generated_text = "";
    for (llama_token tkn : tokens_predicted)
    {
      generated_text += llama_token_to_str(ctx, tkn);
    }
    return std::string(generated_text);
  }

  bool loadPrompt(std::string prompt)
  {
    // prompt.insert(0, " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"); // always add a first space
    prompt.insert(0, 1, ' '); // always add a first space
    std::vector<llama_token> prompt_tokens = ::llama_tokenize(ctx, prompt, true);
    // compare the evaluated prompt with the new prompt
    int new_prompt_len = 0;
    for (int i = 0; i < prompt_tokens.size(); i++)
    {
      if (i < processed_tokens.size() &&
          processed_tokens[i] == prompt_tokens[i])
      {
        continue;
      }
      else
      {
        embd_inp.push_back(prompt_tokens[i]);
        if (new_prompt_len == 0)
        {
          if (i - 1 < n_past)
          {
            processed_tokens.erase(processed_tokens.begin() + i, processed_tokens.end());
          }
          // Evaluate the new fragment prompt from the last token processed.
          n_past = processed_tokens.size();
        }
        new_prompt_len++;
      }
    }
    if (n_past > 0 && params.interactive)
    {
      n_remain -= new_prompt_len;
    }
    fprintf(stderr, "embd_inp size %d,%d\n", embd_inp.size(), params.n_ctx);
    has_next_token = true;
    return true;
  }

  void beginCompletion()
  {
    if (n_remain == 0)
    {
      // number of tokens to keep when resetting context
      if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size())
      {
        params.n_keep = (int)embd_inp.size();
      }
    }
    n_remain = params.n_predict;
  }

  llama_token nextToken()
  {
    llama_token result = -1;
    // fprintf(stderr, "embed size %d,%d,%d,%d\n", embd.size(), embd_inp.size(), n_consumed,n_past);
    if (embd.size() > 0)
    {
      if (n_past + (int)embd.size() > params.n_ctx)
      {
        // Reset context
        const int n_left = n_past - params.n_keep;
        n_past = std::max(1, params.n_keep);
        processed_tokens.erase(processed_tokens.begin() + n_past, processed_tokens.end());
        embd.insert(embd.begin(), last_n_tokens.begin() + params.n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());
      }
      for (int i = 0; i < (int)embd.size(); i += params.n_batch)
      {
        int n_eval = (int)embd.size() - i;
        if (n_eval > params.n_batch)
        {
          n_eval = params.n_batch;
        }
        if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads))
        {
          fprintf(stderr, "%s : failed to eval\n", __func__);
          has_next_token = false;
          return result;
        }
        n_past += n_eval;
      }
    }
    embd.clear();
    if ((int)embd_inp.size() <= n_consumed && has_next_token)
    {
      // out of user input, sample next token
      const float temp = params.temp;
      const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
      const float top_p = params.top_p;
      const float tfs_z = params.tfs_z;
      const float typical_p = params.typical_p;
      const int32_t repeat_last_n = params.repeat_last_n < 0 ? params.n_ctx : params.repeat_last_n;
      const float repeat_penalty = params.repeat_penalty;
      const float alpha_presence = params.presence_penalty;
      const float alpha_frequency = params.frequency_penalty;
      const int mirostat = params.mirostat;
      const float mirostat_tau = params.mirostat_tau;
      const float mirostat_eta = params.mirostat_eta;
      const bool penalize_nl = params.penalize_nl;
      llama_token id = 0;
      {
        auto logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);

        // Apply params.logit_bias map
        for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++)
        {
          logits[it->first] += it->second;
        }

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++)
        {
          candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Apply penalties
        float nl_logit = logits[llama_token_nl()];
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), params.n_ctx);
        llama_sample_repetition_penalty(ctx, &candidates_p,
                                        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                        last_n_repeat, repeat_penalty);
        llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                      last_n_repeat, alpha_frequency, alpha_presence);
        if (!penalize_nl)
        {
          logits[llama_token_nl()] = nl_logit;
        }

        if (temp <= 0)
        {
          // Greedy sampling
          id = llama_sample_token_greedy(ctx, &candidates_p);
        }
        else
        {
          if (mirostat == 1)
          {
            static float mirostat_mu = 2.0f * mirostat_tau;
            const int mirostat_m = 100;
            llama_sample_temperature(ctx, &candidates_p, temp);
            id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
          }
          else if (mirostat == 2)
          {
            static float mirostat_mu = 2.0f * mirostat_tau;
            llama_sample_temperature(ctx, &candidates_p, temp);
            id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
          }
          else
          {
            // Temperature sampling
            llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
            llama_sample_typical(ctx, &candidates_p, typical_p, 1);
            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
            llama_sample_temperature(ctx, &candidates_p, temp);
            id = llama_sample_token(ctx, &candidates_p);
          }
        }
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        processed_tokens.push_back(id);
        num_tokens_predicted++;
      }

      // replace end of text token with newline token when in interactive mode
      if (id == llama_token_eos() && params.interactive)
      {
        id = llama_token_newline.front();
        if (params.antiprompt.size() != 0)
        {
          // tokenize and inject first reverse prompt
          const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
          embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
        }
      }

      // add it to the context
      embd.push_back(id);
      for (auto id : embd)
      {
        result = id;
      }
      // decrement remaining sampling budget
      --n_remain;
    }
    else
    {
      // some user input remains from prompt or interaction, forward it to processing
      while ((int)embd_inp.size() > n_consumed)
      {
        embd.push_back(embd_inp[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[n_consumed]);
        processed_tokens.push_back(embd_inp[n_consumed]);
        ++n_consumed;
        if ((int)embd.size() >= params.n_batch)
        {
          break;
        }
      }
    }
    if (params.interactive && (int)embd_inp.size() <= n_consumed)
    {
      // check for reverse prompt
      if (params.antiprompt.size())
      {
        std::string last_output;
        for (auto id : last_n_tokens)
        {
          last_output += llama_token_to_str(ctx, id);
        }
        has_next_token = true;
        // Check if each of the reverse prompts appears at the end of the output.
        for (std::string &antiprompt : params.antiprompt)
        {
          if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
          {
            has_next_token = false;
            return result;
          }
        }
      }
      if (n_past > 0)
      {
        has_next_token = true;
      }
    }

    if (!embd.empty() && embd.back() == llama_token_eos())
    {
      has_next_token = false;
    }
    else
    {
      if (params.interactive && n_remain <= 0 && params.n_predict != -1)
      {
        n_remain = params.n_predict;
      }
      has_next_token = n_remain != 0;
    }
    return result;
  }

  std::string tokenToString(llama_token token)
  {
    if (token == llama_token_eos())
    {
      return "";
    }
    else if (token == llama_token_nl())
    {
      return "\n";
    }
    else
    {
      return std::string(llama_token_to_str(ctx, token));
    }
  }

  void printInfo()
  {
    fprintf(stderr, "embed size: %d\n", embd.size());
    fprintf(stderr, "embd_inp size: %d\n", embd_inp.size());
  }

private:
  gpt_params params;
  llama_context *ctx;
  llama_context *ctx_for_embedding;
  int threads;
  int n_ctx;

  // std::vector<llama_token> last_n_tokens;
  // std::vector<llama_token> llama_token_newline;
};

// Logic and data behind the server's behavior.
class LlamaServiceImpl final : public LlamaGoService::CallbackService
{

  class Reactor : public grpc::ServerWriteReactor<Output>
  {
  public:
    Reactor(CallbackServerContext *ctx, LlamaServerContext *llama, const Job *request)
        : ctx_(ctx), request_(request), llama_(llama)
    {
      if (llama->loadPrompt(request->prompt()))
      {
        llama->beginCompletion();
        NextWrite();
      }
      else
      {
        Finish(grpc::Status::OK);
      }
    }
    void OnDone() override
    {
      fprintf(stderr, "completion done\n");
      delete this;
    }
    void OnWriteDone(bool /*ok*/) override
    {
      // fprintf(stderr, "on write done");
      NextWrite();
    }
    void OnCancel() override
    {
      FinishOnce(grpc::Status::OK);
    }

  private:
    CallbackServerContext *const ctx_;
    LlamaServerContext *llama_;
    const Job *const request_;
    int n_remain{0};
    std::mutex finish_mu_;
    bool finished_{false};
    Output *response;

    void NextWrite()
    {
      response = new Output();
      // loop inference until finish completion
      if (llama_->has_next_token)
      {
        std::lock_guard<std::mutex> l(finish_mu_);
        auto result = llama_->doCompletion();
        fprintf(stderr, "%s", result.c_str());
        response->set_status(llama::Status::RUNNING);
        response->set_output(result);
        StartWrite(response);
      }
      else
      {
        {
          std::lock_guard<std::mutex>
              l(finish_mu_);
          if (!finished_)
          {
            response->set_status(llama::Status::FINISHED);
            StartWriteLast(response, grpc::WriteOptions());
          }
        }
        // If we use WriteLast, we shouldn't wait before attempting Finish
        FinishOnce(Status::OK);
      }
    }

    void FinishOnce(const Status &s)
    {
      std::lock_guard<std::mutex> l(finish_mu_);
      if (!finished_)
      {
        Finish(s);
        finished_ = true;
      }
    }
  };

public:
  LlamaServiceImpl(LlamaServerContext *llama_) : llama(llama_)
  {
  }

  ServerWriteReactor<Output> *Answer(
      CallbackServerContext *context, const Job *request)
  {
    fprintf(stderr, "%s : new answer request: %s\n", __func__, request->prompt().c_str());
    llama->rewind();
    // std::vector<float> embeded = llama->complete(request->prompt());
    Reactor *reactor = new Reactor(context, llama, request);
    // reactors.push_back(reactor);

    return reactor;
  }

  ServerUnaryReactor *Embed(
      CallbackServerContext *context, const Job *request, Output *response)
  {
    fprintf(stderr, "%s : get embed %s\n", __func__, request->prompt().c_str());
    std::vector<float> embeded = llama->embedding(request->prompt());
    *response->mutable_embed() = {embeded.begin(), embeded.end()};
    response->set_id(request->id());
    ServerUnaryReactor *reactor = context->DefaultReactor();
    reactor->Finish(Status::OK);
    return reactor;
  }

private:
  LlamaServerContext *llama;
  // std::vector<Reactor> reactors;
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

// auto server_params_parse(server_params &sparams)
// {

//   return set_extra_params;
// }

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
  // params.embedding = true;

  sparams.port = 8080;

  std::vector<std::string> extra_args = {"--server", "--port"};

  if (server_params_parse(argc, argv, sparams, params) == false)
  {
    return 1;
  }

  if (gpt_params_parse_with_extra_check(argc, argv, params, &extra_args) == false)
  {
    return 1;
  }

  // params.embedding = true;

  if (params.seed <= 0)
  {
    params.seed = time(NULL);
  }

  fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

  LlamaServerContext llama(params);

  // load the model
  if (!llama.loaded)
  {
    fprintf(stderr, "error: failed to load model\n");
    return 1;
  }

  RunServer(sparams.port, &llama);
  return 0;
}
