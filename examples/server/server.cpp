#include <httplib.h>
#include <json.hpp>
#include "common.h"
#include "llama.h"

struct server_params
{
  std::string hostname = "127.0.0.1";
  int32_t port = 8080;
  int32_t read_timeout = 600;
  int32_t write_timeout = 600;
  bool verbose = false;
};

struct llama_server_context
{
  bool stream = false;
  bool has_next_token = false;
  std::string generated_text = "";

  size_t num_tokens_predicted = 0;
  size_t n_past = 0;
  size_t n_consumed = 0;
  size_t n_session_consumed = 0;
  size_t n_remain = 0;

  std::vector<llama_token> embd;
  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> processed_tokens;
  std::vector<llama_token> embd_inp;

  std::vector<llama_token> last_prompt_tokens;
  llama_context *ctx;
  gpt_params params;

  std::string stopping_word;

  bool verbose = false;
  int json_indent = -1;

  void rewind() {
    params.antiprompt.clear();
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(params.n_ctx);
    stopping_word = "";

    //processed_tokens.clear();
    embd_inp.clear();
    n_remain = 0;
    n_past = 0;
    n_consumed = 0;
  }

  bool loadModel(const gpt_params &params_)
  {
    params = params_;
    ctx = llama_init_from_gpt_params(params);
    if (ctx == NULL)
    {
      fprintf(stderr, "%s: error: unable to load model\n", __func__);
      return false;
    }

    last_n_tokens.resize(params.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    return true;
  }

  bool loadPrompt() {
    params.prompt.insert(0, 1, ' '); // always add a first space
    std::vector<llama_token> prompt_tokens = ::llama_tokenize(ctx, params.prompt, true);
    if (prompt_tokens == last_prompt_tokens)
    {
        embd.clear();
    }
    // compare the evaluated prompt with the new prompt
    for (n_past = 0; n_past < prompt_tokens.size() - 1 && n_past < processed_tokens.size(); n_past++) {
      if (prompt_tokens[n_past] != processed_tokens[n_past]) {
        break;
      }
    }
    processed_tokens.resize(n_past);
    if (prompt_tokens.size() > n_past) {
      embd_inp.insert(embd_inp.end(), prompt_tokens.begin() + n_past, prompt_tokens.end());
    }
    last_prompt_tokens = prompt_tokens;
    has_next_token = true;
    return true;
  }

  void beginCompletion()
  {
    if(n_remain == 0) {
      // number of tokens to keep when resetting context
      if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size())
      {
        params.n_keep = (int)embd_inp.size();
      }
    }
    n_remain = params.n_predict;
    llama_set_rng_seed(ctx, params.seed);
  }

  llama_token nextToken() {
    llama_token result = -1;
    if (embd.size() > 0)
    {
      if (n_past + embd.size() > (size_t)params.n_ctx)
      {
        // Reset context
        const int n_left = n_past - params.n_keep;
        n_past = std::max(1, params.n_keep);
        //processed_tokens.erase(processed_tokens.begin() + n_past, processed_tokens.end());
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
    if (embd_inp.size() <= n_consumed)
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
            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
            llama_sample_temperature(ctx, &candidates_p, temp);
            id = llama_sample_token(ctx, &candidates_p);
          }
        }
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        processed_tokens.push_back(id);
        num_tokens_predicted++;
      }

      // add it to the context
      embd.push_back(id);
      result = id;
      // decrement remaining sampling budget
      --n_remain;
    }
    else
    {
      // some user input remains from prompt or interaction, forward it to processing
      while (embd_inp.size() > n_consumed)
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

    if (!embd.empty() && embd.back() == llama_token_eos()) {
        stopping_word = llama_token_to_str(ctx, embd.back());
        has_next_token = false;
        if (verbose) {
            fprintf(stderr, "eos token found!\n");
        }
        return result;
    }

    has_next_token = params.n_predict == -1 ? true : n_remain != 0;
    return result;
  }

  std::string doCompletion()
  {
    llama_token token = nextToken();
    if (token == -1) {
      return "";
    }

    std::string token_text = llama_token_to_str(ctx, token);
    generated_text += token_text;

    if (verbose) {
      fprintf(stderr,
              "next token: {\n"
              "    token: %d,\n"
              "    token_text: \"%s\",\n"
              "    has_next_token: %d,\n"
              "    n_remain: %ld,\n"
              "    num_tokens_predicted: %ld,\n"
              "    stopping_word: \"%s\",\n"
              "}\n",
              token, token_text.c_str(), has_next_token, n_remain, num_tokens_predicted,
              stopping_word.c_str());
    }

    for (const std::string& word : params.antiprompt) {
      size_t i = generated_text.find(word, generated_text.size() - (word.size() + token_text.size()));
      if (i != std::string::npos) {
        generated_text.erase(generated_text.begin() + i, generated_text.end());
        stopping_word = word;
        has_next_token = false;
        break;
      }
    }

    return token_text;
  }

  std::vector<float> embedding(std::string content, int threads) {
    content.insert(0, 1, ' ');
    std::vector<llama_token> tokens = ::llama_tokenize(ctx, content, true);
    if (tokens.size() > 0)
    {
      if (llama_eval(ctx, tokens.data(), tokens.size(), 0, threads))
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
};

using namespace httplib;

using json = nlohmann::json;

void server_print_usage(int /*argc*/, char **argv, const gpt_params &params, const server_params &sparams)
{
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  -v, --verbose         verbose output (default: false)\n");
  fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
  fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
  fprintf(stderr, "  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
  fprintf(stderr, "  --memory-f32          use f32 instead of f16 for memory key+value (default: disabled)\n");
  fprintf(stderr, "                        not recommended: doubles context memory required and no measurable increase in quality\n");
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
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
  fprintf(stderr, "  -ngl N, --n-gpu-layers N\n");
  fprintf(stderr, "                        number of layers to store in VRAM\n");
#endif
  fprintf(stderr, "  -m FNAME, --model FNAME\n");
  fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
  fprintf(stderr, "  -a ALIAS, --alias ALIAS\n");
  fprintf(stderr, "                        set an alias for the model, will be added as `model` field in completion response\n");
  fprintf(stderr, "  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
  fprintf(stderr, "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
  fprintf(stderr, "  --host                ip address to listen (default  (default: %s)\n", sparams.hostname.c_str());
  fprintf(stderr, "  --port PORT           port to listen (default  (default: %d)\n", sparams.port);
  fprintf(stderr, "  -to N, --timeout N    server read/write timeout in seconds (default: %d)\n", sparams.read_timeout);
  fprintf(stderr, "\n");
}

bool server_params_parse(int argc, char **argv, server_params &sparams, gpt_params &params)
{
  gpt_params default_params;
  server_params default_sparams;
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
    else if (arg == "--timeout" || arg == "-to")
    {
        if (++i >= argc) {
            invalid_param = true;
            break;
        }
        sparams.read_timeout = std::stoi(argv[i]);
        sparams.write_timeout = std::stoi(argv[i]);
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
    else if (arg == "-a" || arg == "--alias")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.model_alias = argv[i];
    }
    else if (arg == "--embedding")
    {
      params.embedding = true;
    }
    else if (arg == "-h" || arg == "--help")
    {
      server_print_usage(argc, argv, default_params, default_sparams);
      exit(0);
    }
    else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.n_ctx = std::stoi(argv[i]);
    }
    else if (arg == "--memory-f32" || arg == "--memory_f32")
    {
      params.memory_f16 = false;
    }
    else if (arg == "--threads" || arg == "-t")
    {
        if (++i >= argc) {
            invalid_param = true;
            break;
        }
        params.n_threads = std::stoi(argv[i]);
    }
    else if (arg == "-b" || arg == "--batch-size")
    {
        if (++i >= argc) {
            invalid_param = true;
            break;
        }
        params.n_batch = std::stoi(argv[i]);
        params.n_batch = std::min(512, params.n_batch);
    }
    else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
      params.n_gpu_layers = std::stoi(argv[i]);
#else
      fprintf(stderr, "warning: not compiled with GPU offload support, --n-gpu-layers option will be ignored\n");
      fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
#endif
    }
    else if (arg == "--lora")
    {
        if (++i >= argc)
        {
            invalid_param = true;
            break;
        }
        params.lora_adapter = argv[i];
        params.use_mmap = false;
    }
    else if (arg == "--lora-base")
    {
        if (++i >= argc) {
            invalid_param = true;
            break;
        }
        params.lora_base = argv[i];
    } else if (arg == "-v" || arg == "--verbose") {
        sparams.verbose = true;
    }
    else
    {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      server_print_usage(argc, argv, default_params, default_sparams);
      exit(1);
    }
  }

  if (invalid_param)
  {
    fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
    server_print_usage(argc, argv, default_params, default_sparams);
    exit(1);
  }
  return true;
}

bool parse_options_completion(json body, llama_server_context& llama, Response &res) {
  gpt_params default_params;
  if (!body["stream"].is_null())
  {
      llama.stream = body["stream"].get<bool>();
  }
  else
  {
      llama.stream = false;
  }
  if (!body["n_predict"].is_null())
  {
      llama.params.n_predict = body["n_predict"].get<int>();
  }
  else
  {
      llama.params.n_predict = default_params.n_predict;
  }
  if (!body["top_k"].is_null())
  {
      llama.params.top_k = body["top_k"].get<int>();
  }
  else
  {
      llama.params.top_k = default_params.top_k;
  }
  if (!body["top_p"].is_null())
  {
      llama.params.top_p = body["top_p"].get<float>();
  }
  else
  {
      llama.params.top_p = default_params.top_p;
  }
  if (!body["tfs_z"].is_null())
  {
      llama.params.tfs_z = body["tfs_z"].get<float>();
  }
  else
  {
      llama.params.tfs_z = default_params.tfs_z;
  }
  if (!body["typical_p"].is_null())
  {
      llama.params.typical_p = body["typical_p"].get<float>();
  }
  else
  {
      llama.params.typical_p = default_params.typical_p;
  }
  if (!body["repeat_last_n"].is_null())
  {
      llama.params.repeat_last_n = body["repeat_last_n"].get<int>();
  }
  else
  {
      llama.params.repeat_last_n = default_params.repeat_last_n;
  }
  if (!body["temperature"].is_null())
  {
      llama.params.temp = body["temperature"].get<float>();
  }
  else
  {
      llama.params.temp = default_params.temp;
  }
  if (!body["repeat_penalty"].is_null())
  {
      llama.params.repeat_penalty = body["repeat_penalty"].get<float>();
  }
  else
  {
      llama.params.repeat_penalty = default_params.repeat_penalty;
  }
  if (!body["presence_penalty"].is_null())
  {
      llama.params.presence_penalty = body["presence_penalty"].get<float>();
  }
  else
  {
      llama.params.presence_penalty = default_params.presence_penalty;
  }
  if (!body["frequency_penalty"].is_null())
  {
      llama.params.frequency_penalty = body["frequency_penalty"].get<float>();
  }
  else
  {
      llama.params.frequency_penalty = default_params.frequency_penalty;
  }
  if (!body["mirostat"].is_null())
  {
      llama.params.mirostat = body["mirostat"].get<float>();
  }
  else
  {
      llama.params.mirostat = default_params.mirostat;
  }
  if (!body["mirostat_tau"].is_null())
  {
      llama.params.mirostat_tau = body["mirostat_tau"].get<float>();
  }
  else
  {
      llama.params.mirostat_tau = default_params.mirostat_tau;
  }
  if (!body["mirostat_eta"].is_null())
  {
      llama.params.mirostat_eta = body["mirostat_eta"].get<float>();
  }
  else
  {
      llama.params.mirostat_eta = default_params.mirostat_eta;
  }
  if (!body["penalize_nl"].is_null())
  {
      llama.params.penalize_nl = body["penalize_nl"].get<float>();
  }
  else
  {
      llama.params.penalize_nl = default_params.penalize_nl;
  }
  if (!body["n_keep"].is_null())
  {
      llama.params.n_keep = body["n_keep"].get<int>();
  }
  else
  {
      llama.params.n_keep = default_params.n_keep;
  }
  if (!body["seed"].is_null())
  {
    llama.params.seed = body["seed"].get<int>();
  }
  else
  {
    llama.params.seed = time(NULL);
  }
  if (!body["ignore_eos"].is_null() && body["ignore_eos"].get<bool>())
  {
      llama.params.logit_bias[llama_token_eos()] = -INFINITY;
  }
  else
  {
      llama.params.logit_bias.erase(llama_token_eos());
  }
  if (!body["prompt"].is_null())
  {
    llama.params.prompt = body["prompt"].get<std::string>();
  }
  else
  {
    json data = {
        {"status", "error"},
        {"reason", "You need to pass the prompt"}};
    res.set_content(data.dump(llama.json_indent), "application/json");
    res.status = 400;
    return false;
  }
  if (!body["stop"].is_null())
  {
    llama.params.antiprompt = body["stop"].get<std::vector<std::string>>();
  }
  else
  {
      llama.params.antiprompt.clear();
  }

  if (llama.verbose) {
      std::string tmp_stop =
          std::accumulate(llama.params.antiprompt.begin(), llama.params.antiprompt.end(),
                          std::string{}, [](std::string a, std::string b) {
                              return a + (a != "" ? ", \"" : "\"") + b + "\"";
                          });

      fprintf(stderr,
              "-------------------------\n"
              "/completion parameters: {\n"
              "    stream: %d,\n"
              "    frequency_penalty: %f,\n"
              "    mirostat: %d,\n"
              "    mirostat_eta: %f,\n"
              "    mirostat_tau: %f,\n"
              "    n_keep: %d,\n"
              "    n_predict: %d,\n"
              "    penalize_nl: %d,\n"
              "    presence_penalty: %f,\n"
              "    repeat_last_n: %d,\n"
              "    repeat_penalty: %f,\n"
              "    seed: %d,\n"
              "    stop: [%s],\n"
              "    temperature: %f,\n"
              "    tfs_z: %f,\n"
              "    top_k: %d,\n"
              "    top_p: %f,\n"
              "    typical_p: %f,\n"
              "}\nPROMPT[%s]\n",
              llama.stream, llama.params.frequency_penalty, llama.params.mirostat,
              llama.params.mirostat_eta, llama.params.mirostat_tau, llama.params.n_keep,
              llama.params.n_predict, llama.params.penalize_nl,
              llama.params.presence_penalty, llama.params.repeat_last_n,
              llama.params.repeat_penalty, llama.params.seed, tmp_stop.c_str(),
              llama.params.temp, llama.params.tfs_z, llama.params.top_k,
              llama.params.top_p, llama.params.typical_p, llama.params.prompt.c_str());
  }

  return true;
}

json format_generation_settings(const llama_server_context& llama) {
  return json {
    { "seed", llama.params.seed },
    { "temp", llama.params.temp },
    { "top_k", llama.params.top_k },
    { "top_p", llama.params.top_p },
    { "tfs_z", llama.params.tfs_z },
    { "typical_p", llama.params.typical_p },
    { "repeat_last_n", llama.params.repeat_last_n },
    { "repeat_penalty", llama.params.repeat_penalty },
    { "presence_penalty", llama.params.presence_penalty },
    { "frequency_penalty", llama.params.frequency_penalty },
    { "mirostat", llama.params.mirostat },
    { "mirostat_tau", llama.params.mirostat_tau },
    { "mirostat_eta", llama.params.mirostat_eta },
    { "penalize_nl", llama.params.penalize_nl }
  };
}

std::string log(const Request &req, const Response &res)
{
    std::string s;

    s += "============ REQUEST ===========\n";
    s += "< ";
    s += req.method;
    s += " ";
    s += req.path;
    s += " ";
    s += req.version;
    s += "\n";

    if (!req.body.empty()) {
        std::string line;
        std::istringstream stream(req.body);
        while (std::getline(stream, line)) {
            s += "< " + line + "\n";
        }
    }

    s += "------------ RESPONSE ------------\n> ";
    s += res.version;
    s += " ";
    s += std::to_string(res.status);
    s += "\n";

    if (!res.body.empty()) {
        std::string line;
        std::istringstream stream(res.body);
        while (std::getline(stream, line)) {
            s += "> " + line + "\n";
        }
    }

    return s;
}

int main(int argc, char **argv)
{
  // own arguments required by this example
  gpt_params params;
  server_params sparams;

  // struct that contains llama context and inference
  llama_server_context llama;
  params.model = "ggml-model.bin";

  if (server_params_parse(argc, argv, sparams, params) == false)
  {
    return 1;
  }

  llama.verbose = sparams.verbose;
  llama.json_indent = sparams.verbose ? 4 : -1;

  if (params.model_alias == "unknown") {
    params.model_alias = params.model;
  }

  // load the model
  if (!llama.loadModel(params))
  {
    return 1;
  }

  Server svr;

  svr.Get("/", [](const Request &, Response &res)
          { res.set_content("<h1>llama.cpp server works</h1>", "text/html"); });

  svr.Post("/completion", [&llama](const Request &req, Response &res) {
      if (llama.params.embedding) {
          json data = {
              {"status", "error"},
              {"reason", "To use completion function, disable embedding mode"}};
          res.set_content(
              data.dump(llama.json_indent, ' ', false, json::error_handler_t::replace),
              "application/json");
          res.status = 400;
          return;
      }

      llama.rewind();

      if (parse_options_completion(json::parse(req.body), llama, res) == false) {
          return;
      }

      if (!llama.loadPrompt()) {
          json data = {{"status", "error"}, {"reason", "Context too long."}};
          res.set_content(
              data.dump(llama.json_indent, ' ', false, json::error_handler_t::replace),
              "application/json");
          res.status = 400;
          return;
      }

      llama.beginCompletion();

      if (!llama.stream) {
          while (llama.has_next_token) {
              llama.doCompletion();
          }

          json data = {{"content", llama.generated_text},
                       {"stop", true},
                       {"model", llama.params.model_alias},
                       {"tokens_predicted", llama.num_tokens_predicted},
                       {"generation_settings", format_generation_settings(llama)},
                       {"prompt", llama.params.prompt},
                       {"stopping_word", llama.stopping_word}};
          return res.set_content(
              data.dump(llama.json_indent, ' ', false, json::error_handler_t::replace),
              "application/json");
      } else {
          const auto chunked_content_provider = [&](size_t, DataSink &sink) {
              while (llama.has_next_token) {
                  std::string token_text = llama.doCompletion();

                  json data;
                  if (llama.has_next_token) {
                      data = {{"content", token_text}, {"stop", false}};
                  } else {
                      // Generation is done, send extra information.
                      data = {
                          {"content", token_text},
                          {"stop", true},
                          {"model", llama.params.model_alias},
                          {"tokens_predicted", llama.num_tokens_predicted},
                          {"generation_settings", format_generation_settings(llama)},
                          {"prompt", llama.params.prompt},
                          {"stopping_word", llama.stopping_word},
                          {"generated_text", llama.generated_text}};
                  }

                  std::string str = "data: " +
                                    data.dump(llama.json_indent, ' ', false,
                                              json::error_handler_t::replace) +
                                    "\n\n";
                  if (!sink.write(str.data(), str.size())) {
                      if (llama.verbose) {
                          fprintf(stderr, "stream closed\n");
                      }
                      return false;
                  }
              }

              sink.done();
              return true;
          };
          res.set_chunked_content_provider("text/event-stream", chunked_content_provider);
      }
  });


  svr.Post("/tokenize", [&llama](const Request &req, Response &res)
            {
              json body = json::parse(req.body);
              json data = {
                    {"tokens", ::llama_tokenize(llama.ctx, body["content"].get<std::string>(), false) } };
                return res.set_content(data.dump(llama.json_indent), "application/json");
            });

  svr.Post("/embedding", [&llama](const Request &req, Response &res)
            {
              if(!llama.params.embedding) {
                std::vector<float> empty;
                json data = {
                    {"embedding", empty}};
                fprintf(stderr, "[llama-server] : You need enable embedding mode adding: --embedding option\n");
                return res.set_content(data.dump(llama.json_indent), "application/json");
              }
              json body = json::parse(req.body);
              std::string content = body["content"].get<std::string>();
              int threads = body["threads"].get<int>();
              json data = {
                    {"embedding", llama.embedding(content, threads) } };
              return res.set_content(data.dump(llama.json_indent), "application/json");
            });

  if(params.embedding) {
    fprintf(stderr, "NOTE: Mode embedding enabled. Completion function doesn't work in this mode.\n");
  }

  if (llama.verbose) {
      svr.set_logger([](const Request &req, const Response &res) {
          fprintf(stderr, "%s", log(req, res).c_str());
      });
  }

  svr.set_exception_handler([](const Request &, Response &res, std::exception_ptr ep) {
      auto fmt = "500 Internal Server Error\n%s";
      char buf[BUFSIZ];
      try {
          std::rethrow_exception(ep);
      } catch (std::exception &e) {
          snprintf(buf, sizeof(buf), fmt, e.what());
      } catch (...) {
          snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
      }
      res.set_content(buf, "text/plain");
      res.status = 500;
  });

  // set timeouts and change hostname and port
  svr.set_read_timeout(sparams.read_timeout);
  svr.set_write_timeout(sparams.write_timeout);

  if (!svr.bind_to_port(sparams.hostname, sparams.port)) {
      fprintf(stderr, "%s: ERROR: couldn't bind server to %s:%i\n", __func__,
              sparams.hostname.c_str(), sparams.port);
      return 1;
  }

  fprintf(stderr, "%s: http server Listening at http://%s:%i\n", __func__,
          sparams.hostname.c_str(), sparams.port);
  if (!svr.listen_after_bind()) {
      return 1;
  }
}
