#include <stdio.h>
#include <iostream>
#include <fstream>

#include "common.h"
#include "llama.h"
#include "crow.h"


auto const BINDPORT = 8001;

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false)
        return 1;

    if (params.n_ctx > 2048)
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);

    if (params.seed <= 0)
        params.seed = time(NULL);

    llama_context * ctx;
    
    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mlock  = params.use_mlock;
        lparams.logits_all = params.perplexity;
        lparams.embedding  = true;
        
        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    crow::SimpleApp app;
    // app.loglevel(crow::LogLevel::Warning);

    /// Python server will send a file name to you.
    /// You should open that file and give the pointer to run_llama.
    /// run_llama will keep writing the output to it.
    /// The python server will keep reading from that file just like it reads
    /// from the stdout of the main process.
    ///
    /// We are doing this because this is probably the simplest way
    /// to get streaming to work here.
    
    CROW_ROUTE(app, "/completion").methods("POST"_method)
    ([&params, &ctx](const crow::request& req){
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(crow::status::BAD_REQUEST);

        // Create new params for this request only
        gpt_params runparams = params;
        
        // Set run params from body
        runparams.prompt         = body["prompt"].s();
        runparams.n_predict      = body["n_predict"].i();
        runparams.top_k          = body["top_k"].i();
        runparams.n_ctx          = body["ctx_size"].i();
        runparams.repeat_last_n  = body["repeat_last_n"].i();
        runparams.top_p          = (float)body["top_p"].d();
        runparams.temp           = (float)body["temp"].d();
        runparams.repeat_penalty = (float)body["repeat_penalty"].d();
        runparams.embedding      = false;

        // Open the tempfile into a stream.
        std::ofstream outfile(body["tempfile"].s(), std::ios::out);

        // Write output of LLaMA to file stream.
        run_llama(ctx, runparams, &outfile);

        return crow::response(crow::status::OK);
    });

    CROW_ROUTE(app, "/embedding").methods("POST"_method)
    ([&params, &ctx](const crow::request& req){
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(crow::status::BAD_REQUEST);
        
        // Create new params for this request only
        gpt_params runparams = params;

        // Set run params from body
        runparams.prompt    = body["prompt"].s();
        runparams.embedding = true;

        // Open the tempfile into a stream.
        std::ofstream outfile(body["tempfile"].s(), std::ios::out);

        // Write output of LLaMA to file stream.
        run_llama_embedding(ctx, runparams, &outfile);

        return crow::response(crow::status::OK);
    });

    app.port(BINDPORT).multithreaded().run();
    return 0;
}