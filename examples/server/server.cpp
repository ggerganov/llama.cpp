#include <server.h>

using namespace httplib;
using namespace json11;

bool Llama::load_context() {
  // load the model
  {
    auto lparams = llama_context_default_params();

    lparams.n_ctx = params.n_ctx;
    lparams.n_parts = params.n_parts;
    lparams.seed = params.seed;
    lparams.f16_kv = params.memory_f16;
    lparams.use_mmap = params.use_mmap;
    lparams.use_mlock = params.use_mlock;

    ctx = llama_init_from_file(params.model.c_str(), lparams);

    if (ctx == NULL)
    {
      fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
      return false;
    }
  }

  n_ctx = llama_n_ctx(ctx);


  // enable interactive mode if reverse prompt or interactive start is specified
  if (params.antiprompt.size() != 0 || params.interactive_first)
  {
    params.interactive = true;
  }

  // determine newline token
  llama_token_newline = ::llama_tokenize(ctx, "\n", false);

  last_n_tokens.resize(n_ctx);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
  printf("Llama Context loaded\n");
  return true;
}

bool Llama::prompt_test() {
  embd_inp = ::llama_tokenize(ctx, params.prompt, true);
  
  if ((int)embd_inp.size() > n_ctx - 4)
  {
    fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
    return false;
  }
  return true;
}

void Llama::setting_context() {
  n_remain   = params.n_predict;

  // number of tokens to keep when resetting context
  if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size() || params.instruct)
  {
    params.n_keep = (int)embd_inp.size();
  }

  // print system information
  {
    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
  }
  
  fprintf(stderr, "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
          params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
  fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
  fprintf(stderr, "\n\n");

  printf("\rLoading prompt: 0%");
 
  while(true) {
    if (embd.size() > 0)
    {
      if (n_past + (int)embd.size() > n_ctx)
      {
        const int n_left = n_past - params.n_keep;
        n_past = params.n_keep;
        embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());
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
          return;
        }
        n_past += n_eval;
      }
    }
    embd.clear();
    if ((int)embd_inp.size() <= n_consumed && !is_interacting)
    {
      // out of user input, sample next token
      const float temp = params.temp;
      const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
      const float top_p = params.top_p;
      const float tfs_z = params.tfs_z;
      const float typical_p = params.typical_p;
      const int32_t repeat_last_n = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
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
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
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
            llama_sample_top_k(ctx, &candidates_p, top_k);
            llama_sample_tail_free(ctx, &candidates_p, tfs_z);
            llama_sample_typical(ctx, &candidates_p, typical_p);
            llama_sample_top_p(ctx, &candidates_p, top_p);
            llama_sample_temperature(ctx, &candidates_p, temp);
            id = llama_sample_token(ctx, &candidates_p);
          }
        }
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      if (id == llama_token_eos() && params.interactive && !params.instruct)
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
        ++n_consumed;
        if ((int)embd.size() >= params.n_batch)
        {
          break;
        }
      }
      printf("\rLoading prompt: %d%%", (int)(((float)n_consumed / (int)embd_inp.size()) * 100));
    }
    if (params.interactive && (int)embd_inp.size() <= n_consumed) {
      // check for reverse prompt
      if (params.antiprompt.size())
      {
        std::string last_output;
        for (auto id : last_n_tokens)
        {
          last_output += llama_token_to_str(ctx, id);
        }
        is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the output.
        for (std::string &antiprompt : params.antiprompt)
        {
          if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
          {
            is_interacting = true;
            is_antiprompt = true;
            context_config = true;
            printf("\rContext prompt and behavior configured.\n");
            return;
          }
        }
      }
    }
  }
  
}

int Llama::set_message(std::string msg) {
  if (msg.length() > 1)
  {
    auto line_inp = ::llama_tokenize(ctx, msg, false);
    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
    n_remain -= line_inp.size();
    printf("\nRequest completion: %i message tokens \n", line_inp.size());
    is_antiprompt = false;
    return line_inp.size();
  } else {
    return 0;
  }
}

std::string Llama::inference() {
   std::string result = "";
   if (embd.size() > 0)
    {
      if (n_past + (int)embd.size() > n_ctx)
      {
        const int n_left = n_past - params.n_keep;
        n_past = params.n_keep;
        embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());
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
          return result;
        }
        n_past += n_eval;
      }
    }
    embd.clear();
    if ((int)embd_inp.size() <= n_consumed && !is_interacting)
    {
      // out of user input, sample next token
      const float temp = params.temp;
      const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
      const float top_p = params.top_p;
      const float tfs_z = params.tfs_z;
      const float typical_p = params.typical_p;
      const int32_t repeat_last_n = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
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
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
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
            llama_sample_top_k(ctx, &candidates_p, top_k);
            llama_sample_tail_free(ctx, &candidates_p, tfs_z);
            llama_sample_typical(ctx, &candidates_p, typical_p);
            llama_sample_top_p(ctx, &candidates_p, top_p);
            llama_sample_temperature(ctx, &candidates_p, temp);
            id = llama_sample_token(ctx, &candidates_p);
          }
        }
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      if (id == llama_token_eos() && params.interactive && !params.instruct)
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
        result += llama_token_to_str(ctx, id);
        tokens_completion++;
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
        ++n_consumed;
        if ((int)embd.size() >= params.n_batch)
        {
          break;
        }
      }
    }
    if (params.interactive && (int)embd_inp.size() <= n_consumed) {
      // check for reverse prompt
      if (params.antiprompt.size())
      {
        std::string last_output;
        for (auto id : last_n_tokens)
        {
          last_output += llama_token_to_str(ctx, id);
        }
        is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the output.
        for (std::string &antiprompt : params.antiprompt)
        {
          if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
          {
            is_interacting = true;
            is_antiprompt = true;
            return result;
          }
        }
      }
      if (n_past > 0)
      {
        is_interacting = false;
      }
    }
    if (params.interactive && n_remain <= 0 && params.n_predict != -1)
    {
      n_remain = params.n_predict;
      is_interacting = true;
    }
    return result;
}

void server_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
    fprintf(stderr, "  --memory_f32          use f32 instead of f16 for memory key+value\n");
    fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
    if (llama_mlock_supported()) {
        fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        fprintf(stderr, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -host                 ip address to listen (default 0.0.0.0)\n");
    fprintf(stderr, "  -port PORT            port to listen (default 8080)\n");
    fprintf(stderr, "\n");
}


int main(int argc, char ** argv) {

  // own arguments required by this example
  gpt_params default_params;

  gpt_params params;
  params.model = "ggml-model.bin";

  std::string hostname = "0.0.0.0";
  int port = 8080;

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
        port = std::stoi(argv[i]);
        }
        else if (arg == "--host")
        {
        if (++i >= argc)
        {
          invalid_param = true;
          break;
        }
        hostname = argv[i];
        }
        else if (arg == "--keep")
        {
        if (++i >= argc)
        {
          invalid_param = true;
          break;
        }
        params.n_keep = std::stoi(argv[i]);
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

  if (params.seed <= 0)
  {
    params.seed = time(NULL);
  }

  fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

  Llama* llama = new Llama(params);

  // load model file
  if(!llama->load_context()) {
    return 1;
  }

  Server svr;

  svr.Get("/", [](const Request &req, Response &res)
          { 
            res.set_content("<h1>llama.cpp server works</h1>", "text/html");
          }
  );

  svr.Post("/setting-context", [&llama](const Request &req, Response &res) {
            if(!llama->context_config){
              std::string err;
              Json body = Json::parse(req.body, err);
              // seed whould be passed by the request, but seem 
              // the current implementation need it in the load file
              if (!body["threads"].is_null())
              {
                llama->params.n_threads = body["threads"].int_value();
              }
              if (!body["n_predict"].is_null())
              {
                llama->params.n_predict = body["n_predict"].int_value();
              }
              if (!body["top_k"].is_null())
              {
                llama->params.top_k = body["top_k"].int_value();
              }
              if (!body["top_p"].is_null())
              {
                llama->params.top_p = body["top_p"].number_value();
              }
              if (!body["temperature"].is_null())
              {
                llama->params.temp = body["temperature"].number_value();
              }
              if (!body["batch_size"].is_null())
              {
                llama->params.n_batch = body["batch_size"].number_value();
              }
              if (!body["context"].is_null())
              {
                llama->params.prompt = "";
                Json::array context_messages = body["context"].array_items();
                // generate prompt for vicuna model v1
                for (Json ctx_msg : context_messages)
                {
                  auto role = ctx_msg["role"].string_value();
                  if (role == "system")
                  {
                    llama->params.prompt = ctx_msg["content"].string_value() + "\n\n";
                  }
                  else if (role == "user")
                  {
                    llama->params.prompt += "### Human: " + ctx_msg["content"].string_value() + "\n";
                  }
                  else if (role == "assistant")
                  {
                    llama->params.prompt += "### Assistant: " + ctx_msg["content"].string_value() + "\n";
                  }
                }
                llama->params.prompt += "### Human:";
              }
              else
              {
                Json data = Json::object{
                    {"status", "error"},
                    {"reason", "Missing Context"}};
                res.set_content(data.dump(), "application/json");
                res.status = 400;
                return;
              }

              if(!llama->prompt_test())
              {
                Json data = Json::object{
                    {"status", "error"},
                    {"reason", "Context too long, please be more specific"}};
                res.set_content(data.dump(), "application/json");
                res.status = 400;
                return;
              }

              // Default configs for interactive with Vicuna model
              llama->params.interactive = true;
              llama->params.antiprompt.push_back("### Human:");
              llama->params.repeat_last_n = 64;
              llama->params.repeat_penalty = 1.1f;
              llama->setting_context();
          }
            Json data = Json::object {
                { "status", "done" }};
            res.set_content(data.dump(), "application/json");
  });

  svr.Post("/set-message", [&llama](const Request &req, Response &res) {
    bool result = false;
    if (llama->context_config)
    {
      std::string err;
      Json body = Json::parse(req.body, err);
      result = llama->set_message(body["message"].string_value() + "\n");
    }
    Json data = Json::object{
        {"can_inference", result }};
    res.set_content(data.dump(), "application/json");
  });

  svr.Get("/completion", [&llama](const Request &req, Response &res)
          { 
            bool stream = false;
            if (req.has_param("stream")) {
                stream = req.get_param_value("stream") == "true";
            }
            if(stream) {
              // Stream token by token like Chat GPT
              res.set_content_provider(
                "application/json",
                [&llama](size_t offset, DataSink &sink)
                {
                  int ignore = 0;
                  llama->tokens_completion = 0;
                  while(!llama->is_antiprompt) {
                    std::string result = llama->inference();
                    // ignore ### Human: and ### Assistant:
                    if(result == "##") {
                      ignore = 5;
                    }
                    if(ignore == 0) {
                      Json data = Json::object{
                          {"content", result.c_str()},
                          {"tokens_consumed", 1 }};
                      std::string json_data = data.dump();
                      sink.write(json_data.c_str(), json_data.length());
                    } else {
                      ignore--;
                    }
                    printf("\rProcessing: %i tokens processed.", llama->tokens_completion);
                  }
                  sink.write("[DONE]", 6);
                  sink.done(); // No more data
                  printf("\rCompletion finished: %i tokens predicted.\n", llama->tokens_completion);
                  return true; // return 'false' if you want to cancel the process.
                });
            } else {
              // Send all completion when finish
              int ignore = 0;
              std::string completion = "";
              llama->tokens_completion = 0;
              while (!llama->is_antiprompt)
              {
                std::string result = llama->inference();
                // ignore ### Human: and ### Assistant:
                if (result == "##")
                {
                  ignore = 5;
                }
                if (ignore == 0)
                {
                  completion += result;
                  printf("\rProcessing: %i tokens processed.", llama->tokens_completion);
                }
                else
                {
                  ignore--;
                }
              }
              Json data = Json::object {
                { "content", completion.c_str() },
                { "total_tokens", llama->tokens_completion }
              };
            printf("\rCompletion finished: %i tokens predicted.\n", llama->tokens_completion);
            res.set_content(data.dump(), "application/json");
            }
  });

  printf("llama.cpp HTTP Server Listening at http://%s:%i", hostname, port);
  
  // change hostname and port
  svr.listen(hostname, port);
}