#include "common.h"
#include "llama.h"

#include <algorithm>
#include <fstream>

struct retrieval_params {
    std::vector<std::string> context_files; // context files to embed
    int32_t chunk_size            = 64;     // chunk size for context embedding
    std::string chunk_separator   = "\n";   // chunk separator for context embedding
};

static void retrieval_params_print_usage(int argc, char ** argv, gpt_params & gpt_params, retrieval_params & params) {
    gpt_params_print_usage(argc, argv, gpt_params);
    printf("retrieval options:\n");
    printf("  --context-file FNAME  file containing context to embed.\n");
    printf("                        specify multiple files by providing --context-file option multiple times.\n");
    printf("  --chunk-size N        minimum length of embedded text chunk (default:%d)\n", params.chunk_size);
    printf("  --chunk-separator STRING\n");
    printf("                        string to separate chunks (default: \"\\n\")\n");
    printf("\n");
}

static void retrieval_params_parse(int argc, char ** argv, gpt_params & gpt_params, retrieval_params & retrieval_params) {
    int i = 1;
    std::string arg;
    while (i < argc) {
        arg = argv[i];
        bool invalid_gpt_param = false;
        if(gpt_params_find_arg(argc, argv, argv[i], gpt_params, i, invalid_gpt_param)) {
            if (invalid_gpt_param) {
                fprintf(stderr, "error: invalid argument: %s\n", arg.c_str());
                retrieval_params_print_usage(argc, argv, gpt_params, retrieval_params);
                exit(1);
            }
            // option was parsed by gpt_params_find_arg
        } else if (arg == "--context-file") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for --context-file\n");
                retrieval_params_print_usage(argc, argv, gpt_params, retrieval_params);
                exit(1);
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                retrieval_params_print_usage(argc, argv, gpt_params, retrieval_params);
                exit(1);
            }
            // store the external file name in params
            retrieval_params.context_files.push_back(argv[i]);
        } else if (arg == "--chunk-size") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for --chunk-size\n");
                retrieval_params_print_usage(argc, argv, gpt_params, retrieval_params);
                exit(1);
            }
            retrieval_params.chunk_size = std::stoi(argv[i]);
        } else if (arg == "--chunk-separator") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for --chunk-separator\n");
                retrieval_params_print_usage(argc, argv, gpt_params, retrieval_params);
                exit(1);
            }
            retrieval_params.chunk_separator = argv[i];
        } else {
            // unknown argument
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            retrieval_params_print_usage(argc, argv, gpt_params, retrieval_params);
            exit(1);
        }
        i++;
    }
}

struct chunk {
    // filename
    std::string filename;
    // original file position
    size_t filepos;
    // original text data
    std::string textdata = "";
    // tokenized text data
    std::vector<llama_token> tokens;
    // embedding
    std::vector<float> embedding;
};

// chunk file data to chunks of size >= chunk_size
// chunk_separator is the separator between chunks
static std::vector<chunk> chunk_file(const std::string & filename, int chunk_size, const std::string & chunk_separator) {
    std::vector<chunk> chunks;
    std::ifstream f(filename.c_str());

    if (!f.is_open()) {
        fprintf(stderr, "Error: could not open file %s\n", filename.c_str());
        return chunks;
    }

    chunk current_chunk;
    char buffer[1024];
    int64_t filepos = 0;
    std::string current = "";
    while (f.read(buffer, 1024)) {
        current += std::string(buffer, f.gcount());
        size_t pos;
        while ((pos = current.find(chunk_separator)) != std::string::npos) {
            current_chunk.textdata += current.substr(0, pos + chunk_separator.size());
            if ((int) current_chunk.textdata.size() > chunk_size) {
                // save chunk
                current_chunk.filepos = filepos;
                current_chunk.filename = filename;
                chunks.push_back(current_chunk);
                // update filepos
                filepos += (int) current_chunk.textdata.size();
                // reset current_chunk
                current_chunk = chunk();
            }
            current = current.substr(pos + chunk_separator.size());
        }

    }
    // add leftover data to last chunk
    if (current_chunk.textdata.size() > 0) {
        if (chunks.empty()) {
            current_chunk.filepos = filepos;
            current_chunk.filename = filename;
            chunks.push_back(current_chunk);
        } else {
            chunks.back().textdata += current_chunk.textdata;
        }
    }
    f.close();
    return chunks;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, int seq_id) {
    for (size_t i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, { seq_id }, i == tokens.size() - 1);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, i);
                continue;
            }
        }

        float * out = output + batch.seq_id[i][0] * n_embd;
        llama_embd_normalize(embd, out, n_embd);
    }
}

int main(int argc, char ** argv) {
    gpt_params params;
    retrieval_params retrieval_params;

    retrieval_params_parse(argc, argv, params, retrieval_params);

    // For BERT models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;

    if (retrieval_params.chunk_size <= 0) {
        fprintf(stderr, "chunk_size must be positive\n");
        return 1;
    }
    if (retrieval_params.context_files.empty()) {
        fprintf(stderr, "context_files must be specified\n");
        return 1;
    }
    params.embedding = true;

    print_build_info();

    printf("processing files:\n");
    for (auto & context_file : retrieval_params.context_files) {
        printf("%s\n", context_file.c_str());
    }

    std::vector<chunk> chunks;
    for (auto & context_file : retrieval_params.context_files) {
        std::vector<chunk> file_chunk = chunk_file(context_file, retrieval_params.chunk_size, retrieval_params.chunk_separator);
        chunks.insert(chunks.end(), file_chunk.begin(), file_chunk.end());
    }
    printf("Number of chunks: %ld\n", chunks.size());

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
    }

    // max batch size
    const uint64_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    // tokenize the prompts and trim
    for (auto & chunk : chunks) {
        auto inp = ::llama_tokenize(ctx, chunk.textdata, true, false);
        if (inp.size() > n_batch) {
            fprintf(stderr, "%s: error: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        // add eos if not present
        if (inp.empty() || inp.back() != llama_token_eos(model)) {
            inp.push_back(llama_token_eos(model));
        }
        chunk.tokens = inp;
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) chunks.size(); i++) {
            fprintf(stderr, "%s: prompt %d: '%s'\n", __func__, i, chunks[i].textdata.c_str());
            fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, chunks[i].tokens.size());
            for (int j = 0; j < (int) chunks[i].tokens.size(); j++) {
                fprintf(stderr, "%6d -> '%s'\n", chunks[i].tokens[j], llama_token_to_piece(ctx, chunks[i].tokens[j]).c_str());
            }
            fprintf(stderr, "\n\n");
        }
    }

    // initialize batch
    const int n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd = llama_n_embd(model);
    std::vector<float> embeddings(n_chunks * n_embd, 0);
    float * emb = embeddings.data();

    // break into batches
    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_chunks; k++) {
        // clamp to n_batch tokens
        auto & inp = chunks[k].tokens;

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float * out = emb + p * n_embd;
            batch_decode(ctx, batch, out, s, n_embd);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + p * n_embd;
    batch_decode(ctx, batch, out, s, n_embd);

    // save embeddings to chunks
    for (int i = 0; i < n_chunks; i++) {
        chunks[i].embedding = std::vector<float>(emb + i * n_embd, emb + (i + 1) * n_embd);
        // clear tokens as they are no longer needed
        chunks[i].tokens.clear();
    }

    // start loop, receive query and return top k similar chunks based on cosine similarity
    std::string query;
    while (true) {
        printf("Enter query: ");
        std::getline(std::cin, query);
        std::vector<int32_t> query_tokens = llama_tokenize(ctx, query, true);

        struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);
        batch_add_seq(query_batch, query_tokens, 0);

        std::vector<float> query_emb(n_embd, 0);
        batch_decode(ctx, query_batch, query_emb.data(), 1, n_embd);

        llama_batch_clear(query_batch);

        // compute cosine similarities
        {
            std::vector<std::pair<int, float>> similarities;
            for (int i = 0; i < n_chunks; i++) {
                float sim = llama_embd_similarity_cos(chunks[i].embedding.data(), query_emb.data(), n_embd);
                similarities.push_back(std::make_pair(i, sim));
            }

            // sort similarities
            std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
                return a.second > b.second;
            });

            printf("Top %d similar chunks:\n", params.sparams.top_k);
            for (int i = 0; i < std::min(params.sparams.top_k, (int) chunks.size()); i++) {
                printf("filename: %s\n", chunks[similarities[i].first].filename.c_str());
                printf("filepos: %lld\n", (long long int) chunks[similarities[i].first].filepos);
                printf("similarity: %f\n", similarities[i].second);
                printf("textdata:\n%s\n", chunks[similarities[i].first].textdata.c_str());
                printf("--------------------\n");
            }
        }
    }

    // clean up
    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}
