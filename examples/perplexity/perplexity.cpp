#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct results_perplexity {
    std::vector<llama_token> tokens;
    double                   ppl_value;
    std::vector<float>       logits;
    std::vector<float>       probs;
};

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const struct results_perplexity & results
) {
    if (params.logdir.empty()) {
        return;
    }

    if (params.hellaswag) {
        fprintf(stderr, "%s: warning: logging results is not implemented for HellaSwag. No files will be written.\n", __func__);
        return;
    }

    const std::string timestamp = get_sortable_timestamp();

    const bool success = create_directory_with_parents(params.logdir);
    if (!success) {
        fprintf(stderr, "%s: warning: failed to create logdir %s, cannot write logfile\n",
                __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: main\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    dump_non_result_info_yaml(logfile, params, ctx, timestamp, results.tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Perplexity Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    dump_vector_float_yaml(logfile, "logits", results.logits);
    fprintf(logfile, "ppl_value: %f\n", results.ppl_value);
    dump_vector_float_yaml(logfile, "probs", results.probs);

    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile);
}

static std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history
) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + i*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static results_perplexity perplexity_v2(llama_context * ctx, const gpt_params & params) {
    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

    fprintf(stderr, "%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);

    const int n_ctx = llama_n_ctx(ctx);

    if (int(tokens.size()) < 2*n_ctx) {
        fprintf(stderr, "%s: you need at least %d tokens to evaluate perplexity with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    logit_history.resize(tokens.size());
    prob_history.resize(tokens.size());

    if (params.ppl_stride <= 0) {
        fprintf(stderr, "%s: stride is %d but must be greater than zero!\n",__func__,params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int calc_chunk = n_ctx;

    fprintf(stderr, "%s: have %zu tokens. Calculation chunk = %d\n", __func__, tokens.size(), calc_chunk);

    if (int(tokens.size()) <= calc_chunk) {
        fprintf(stderr, "%s: there are only %zu tokens, this is not enough for a context size of %d and stride %d\n",__func__,
                tokens.size(), n_ctx, params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int n_chunk_max = (tokens.size() - calc_chunk + params.ppl_stride - 1)  / params.ppl_stride;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;

    fprintf(stderr, "%s: calculating perplexity over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * params.ppl_stride;
        const int end   = start + calc_chunk;

        const int num_batches = (calc_chunk + n_batch - 1) / n_batch;
        //fprintf(stderr, "%s: evaluating %d...%d using %d batches\n", __func__, start, end, num_batches);

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_kv_cache_clear(ctx);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            //fprintf(stderr, "    Batch %d: starts at %d, size is %d, n_past is %d\n",j,batch_start,batch_size,j * n_batch);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + batch_start, batch_size, j * n_batch, 0))) {
                //fprintf(stderr, "%s : failed to eval\n", __func__);
                return {tokens, -1, logit_history, prob_history};
            }

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_token_bos(llama_get_model(ctx));
            }

            const auto batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);

            if (j == 0) {
                tokens[batch_start] = token_org;
            }
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            fprintf(stderr, "%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                fprintf(stderr, "%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            fprintf(stderr, "%.2f minutes\n", total_seconds / 60.0);
        }

        //fprintf(stderr, "%s: using tokens %d...%d\n",__func__,params.n_ctx - params.ppl_stride + start, params.n_ctx + start);
        for (int j = n_ctx - params.ppl_stride - 1; j < n_ctx - 1; ++j) {

            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + (j + 0) * n_vocab,
                logits.begin() + (j + 1) * n_vocab);

            const float prob = softmax(tok_logits)[tokens[start + j + 1]];
            logit_history[start + j + 1] = tok_logits[tokens[start + j + 1]];
            prob_history[start + j + 1]  = prob;

            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        if (params.ppl_output_type == 0) {
            printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        } else {
            printf("%8d  %.4lf\n", i*params.ppl_stride, std::exp(nll / count));
        }
        fflush(stdout);
    }
    printf("\n");

    return {tokens, std::exp(nll / count), logit_history, prob_history};
}

static results_perplexity perplexity(llama_context * ctx, const gpt_params & params) {
    if (params.ppl_stride > 0) {
        return perplexity_v2(ctx, params);
    }

    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    const int n_ctx = llama_n_ctx(ctx);

    auto tim1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);

    auto tim2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (int(tokens.size()) < 2*n_ctx) {
        fprintf(stderr, "%s: you need at least %d tokens to evaluate perplexity with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    std::vector<float> logit_history;
    logit_history.resize(tokens.size());

    std::vector<float> prob_history;
    prob_history.resize(tokens.size());

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;

    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    fprintf(stderr, "%s: calculating perplexity over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_kv_cache_clear(ctx);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_token_bos(llama_get_model(ctx));
            }

            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + batch_start, batch_size, j * n_batch, 0))) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return {tokens, -1, logit_history, prob_history};
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            if (num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
            }
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            fprintf(stderr, "%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                fprintf(stderr, "%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            fprintf(stderr, "%.2f minutes\n", total_seconds / 60.0);
        }

        // We get the logits for all the tokens in the context window (params.n_ctx)
        // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
        // calculate the perplexity over the last half of the window (so the model always has
        // some context to predict the token).
        //
        // We rely on the fact that attention in the forward pass only looks at previous
        // tokens here, so the logits returned for each token are an accurate representation
        // of what the model would have predicted at that point.
        //
        // Example, we have a context window of 512, we will compute perplexity for each of the
        // last 256 tokens.  Then, we split the input up into context window size chunks to
        // process the entire prompt.
        const int first = n_ctx/2;
        const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits(ctx);
        process_logits(n_vocab, all_logits + first*n_vocab, tokens.data() + start + first, n_ctx - 1 - first,
                       workers, nll, nll2, logit_history.data() + start + first, prob_history.data() + start + first);
        count += n_ctx - first - 1;

        // perplexity is e^(average negative log-likelihood)
        if (params.ppl_output_type == 0) {
            printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        } else {
            double av = nll/count;
            double av2 = nll2/count - av*av;
            if (av2 > 0) av2 = sqrt(av2/(count-1));
            printf("%8d  %.4lf  %4lf  %4lf\n", i*n_ctx, std::exp(nll / count), av, av2);
        }
        fflush(stdout);

        logits.clear();
    }
    printf("\n");

    nll2 /= count;
    nll /= count;
    const double ppl = exp(nll);
    nll2 -= nll * nll;
    if (nll2 > 0) {
        nll2 = sqrt(nll2/(count-1));
        printf("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
    } else {
        printf("Unexpected negative standard deviation of log(prob)\n");
    }

    return {tokens, ppl, logit_history, prob_history};
}

static bool decode_helper(llama_context * ctx, llama_batch & batch, std::vector<float> & batch_logits, int32_t n_batch, int32_t n_vocab) {
    for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
        const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
            0, 0, 0, // unused
        };

        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0) {
            LOG_TEE("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
            return false;
        }

        memcpy(batch_logits.data() + i*n_vocab, llama_get_logits(ctx), n_tokens*n_vocab*sizeof(float));
    }

    return true;
}

static void compute_logprobs(const float * batch_logits, int n_vocab, std::vector<std::thread>& workers,
        const std::vector<std::pair<size_t, llama_token>>& eval_pairs, std::vector<float>& eval_results) {
    constexpr int k_token_chunk = 4;
    if (eval_results.size() != eval_pairs.size()) {
        eval_results.resize(eval_pairs.size());
    }
    if (eval_pairs.empty()) return;

    size_t max_threads = std::min((eval_pairs.size() + k_token_chunk - 1)/k_token_chunk, workers.size());

    std::atomic<int> counter(0);
    auto compute = [&counter, &eval_pairs, &eval_results, batch_logits, n_vocab] () {
        float local_logprobs[k_token_chunk];
        while (true) {
            size_t first = counter.fetch_add(k_token_chunk, std::memory_order_relaxed);
            if (first >= eval_results.size()) break;
            size_t last = std::min(first + k_token_chunk, eval_results.size());
            for (size_t i = first; i < last; ++i) {
                auto logits = batch_logits + eval_pairs[i].first * n_vocab;
                float max_logit = logits[0];
                for (int j = 1; j < n_vocab; ++j) {
                    max_logit = std::max(max_logit, logits[j]);
                }
                float sum_p = 0.f;
                for (int j = 0; j < n_vocab; ++j) {
                    sum_p += expf(logits[j] - max_logit);
                }
                local_logprobs[i - first] = logits[eval_pairs[i].second] - max_logit - std::log(sum_p);
            }
            std::memcpy(eval_results.data() + first, local_logprobs, (last - first)*sizeof(float));
        }
    };

    for (size_t it = 0; it < max_threads; ++it) {
        workers[it] = std::thread(compute);
    }
    for (size_t it = 0; it < max_threads; ++it) {
        workers[it].join();
    }

}

static void hellaswag_score(llama_context * ctx, const gpt_params & params) {
    // Calculates hellaswag score (acc_norm) from prompt
    //
    // Data extracted from the HellaSwag validation dataset (MIT license) https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
    // All used data fields are preprocessed as in https://github.com/EleutherAI/lm-evaluation-harness/blob/df3da98c5405deafd519c2ddca52bb7c3fe36bef/lm_eval/tasks/hellaswag.py#L62-L68
    //
    // All 10042 tasks should be extracted to keep the results standardized like other implementations.
    //
    // Datafile layout:
    // ['??'] denotes json fields
    // 6 lines per task:
    // ['activity_label'] + ": " +['ctx']  - The first part of the query, the context
    // ['label'] - The index the best common sense ending aka gold ending
    // ['endings'][0] - Endings added to the first part of the query
    // ['endings'][1]
    // ['endings'][2]
    // ['endings'][3]

    std::vector<std::string> prompt_lines;
    std::istringstream strstream(params.prompt);
    std::string line;

    while (std::getline(strstream,line,'\n')) {
        prompt_lines.push_back(line);
    }

    if (prompt_lines.size() % 6 != 0) {
        fprintf(stderr, "%s : number of lines in prompt not a multiple of 6.\n", __func__);
        return;
    }

    size_t hs_task_count = prompt_lines.size()/6;
    fprintf(stderr, "%s : loaded %zu tasks from prompt.\n", __func__, hs_task_count);

    const bool is_spm = llama_vocab_type(llama_get_model(ctx)) == LLAMA_VOCAB_TYPE_SPM;
    fprintf(stderr, "================================= is_spm = %d\n", is_spm);

    // This is needed as usual for LLaMA models
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

    // Number of tasks to use when computing the score
    if (params.hellaswag_tasks < hs_task_count) {
        hs_task_count = params.hellaswag_tasks;
    }

    // The tasks should be randomized so the score stabilizes quickly.
    bool randomize_tasks = true;

    // The random seed should not impact the final result if the computation is done over enough tasks, so kept hardcoded for now
    std::mt19937 rng(1);

    // Dataholder for hellaswag tasks
    struct hs_data_t {
        std::string context;
        size_t gold_ending_idx;
        std::string ending[4];
        size_t ending_logprob_count[4];
        double ending_logprob[4];

        size_t i_batch;         // starting index in the llama_batch
        size_t common_prefix;   // max number of initial tokens that are the same in all sentences
        size_t required_tokens; // needed number of tokens to evaluate all 4 endings
        std::vector<llama_token> seq_tokens[4];
    };

    fprintf(stderr, "%s : selecting %zu %s tasks.\n", __func__, hs_task_count, (randomize_tasks?"randomized":"the first")  );

    // Select and read data from prompt lines
    std::vector<hs_data_t> hs_data(hs_task_count);
    for (size_t i = 0; i < hs_task_count; i++) {
        size_t idx = i;

        auto & hs_cur = hs_data[i];

        // Select a random example of those left in the prompt
        if (randomize_tasks) {
            std::uniform_int_distribution<size_t> dist(0, prompt_lines.size()/6-1 ) ;
            idx = dist(rng);
        }

        hs_cur.context = prompt_lines[idx*6];
        hs_cur.gold_ending_idx = std::stoi( prompt_lines[idx*6+1] );
        for (size_t j = 0; j < 4; j++) {
            hs_cur.ending[j] = prompt_lines[idx*6+2+j];
            hs_cur.seq_tokens[j] = ::llama_tokenize(ctx, hs_cur.context + " " + hs_cur.ending[j], add_bos);
        }

        // determine the common prefix of the endings
        hs_cur.common_prefix = 0;
        for (size_t k = 0; k < hs_cur.seq_tokens[0].size(); k++) {
            if (hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[1][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[2][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[3][k]) {
                break;
            }
            hs_cur.common_prefix++;
        }
        hs_cur.required_tokens = hs_cur.common_prefix +
            hs_cur.seq_tokens[0].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[1].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[2].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[3].size() - hs_cur.common_prefix;

        //GGML_ASSERT(hs_cur.common_prefix >= ::llama_tokenize(ctx, hs_cur.context, add_bos).size());

        // Delete the selected random example from the prompt
        if (randomize_tasks) {
            prompt_lines.erase( std::next(prompt_lines.begin(),idx*6)  , std::next(prompt_lines.begin(),idx*6+6) );
        }
    }

    fprintf(stderr, "%s : calculating hellaswag score over selected tasks.\n", __func__);

    printf("\ntask\tacc_norm\n");

    double acc = 0.0f;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int max_tasks_per_batch = 32;
    const int max_seq = 4*max_tasks_per_batch;

    llama_batch batch = llama_batch_init(n_ctx, 0, max_seq);

    std::vector<float> tok_logits(n_vocab);
    std::vector<float> batch_logits(n_vocab*n_ctx);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

    for (size_t i0 = 0; i0 < hs_task_count; i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_batch = 0; // this tells us where in `llama_batch` we are currently

        llama_batch_clear(batch);

        // batch as much tasks as possible into the available context
        // each task has 4 unique seuqnce ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all ending tokens of each sequence
        while (n_cur + (int) hs_data[i1].required_tokens <= n_ctx) {
            auto & hs_cur = hs_data[i1];

            const int s0 = 4*(i1 - i0);
            if (s0 + 4 > max_seq) {
                break;
            }

            for (size_t i = 0; i < hs_cur.common_prefix; ++i) {
                llama_batch_add(batch, hs_cur.seq_tokens[0][i], i, { s0 + 0, s0 + 1, s0 + 2, s0 + 3}, false);
            }
            batch.logits[batch.n_tokens - 1] = true; // we need logits for the last token of the common prefix

            for (int s = 0; s < 4; ++s) {
                for (size_t i = hs_cur.common_prefix; i < hs_cur.seq_tokens[s].size(); ++i) {
                    llama_batch_add(batch, hs_cur.seq_tokens[s][i], i, { s0 + s }, true);
                }
            }

            hs_cur.i_batch = i_batch;
            i_batch += hs_cur.required_tokens;

            n_cur += hs_data[i1].required_tokens;
            if (++i1 == hs_task_count) {
                break;
            }
        }

        if (i0 == i1) {
            fprintf(stderr, "%s : task %zu does not fit in the context window\n", __func__, i0);
            return;
        }

        llama_kv_cache_clear(ctx);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            fprintf(stderr, "%s: llama_decode() failed\n", __func__);
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];
            size_t li = hs_cur.common_prefix;
            for (int s = 0; s < 4; ++s) {
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    eval_pairs.push_back(std::make_pair(hs_cur.i_batch + li++, hs_cur.seq_tokens[s][j + 1]));
                }
                ++li;
            }
        }
        // Then we do the actual calculation
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];

            std::memcpy(tok_logits.data(), batch_logits.data() + n_vocab*(hs_cur.i_batch + hs_cur.common_prefix - 1), n_vocab*sizeof(float));

            const auto first_probs = softmax(tok_logits);

            for (int s = 0; s < 4; ++s) {
                hs_cur.ending_logprob_count[s] = 1;
                hs_cur.ending_logprob[s] = std::log(first_probs[hs_cur.seq_tokens[s][hs_cur.common_prefix]]);
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    hs_cur.ending_logprob[s] += eval_results[ir++];
                    hs_cur.ending_logprob_count[s]++;
                }
                hs_cur.ending_logprob[s] /= hs_cur.ending_logprob_count[s];
            }

            // Find the ending with maximum logprob
            size_t ending_logprob_max_idx = 0;
            double ending_logprob_max_val = hs_cur.ending_logprob[0];
            for (size_t s = 1; s < 4; s++) {
                if (hs_cur.ending_logprob[s] > ending_logprob_max_val) {
                    ending_logprob_max_idx = s;
                    ending_logprob_max_val =  hs_cur.ending_logprob[s];
                }
            }

            //printf("max logprob ending idx %lu, gold ending idx %lu\n", ending_logprob_max_idx, hs_cur.gold_ending_idx);

            // If the gold ending got the maximum logprobe add one accuracy point
            if (ending_logprob_max_idx == hs_cur.gold_ending_idx) {
                acc += 1.0;
            }

            // Print the accumulated accuracy mean x 100
            printf("%zu\t%.8lf\n", i + 1, acc/double(i + 1)*100.0);
            fflush(stdout);
        }

        i0 = i1 - 1;
    }

    llama_batch_free(batch);

    printf("\n");
}

struct winogrande_entry {
    std::string first;
    std::string second;
    std::array<std::string, 2> choices;
    int answer;

    size_t i_batch;
    size_t common_prefix;
    size_t required_tokens;
    size_t n_base1; // number of tokens for context + choice 1
    size_t n_base2; // number of tokens for context + choice 2
    std::vector<llama_token> seq_tokens[2];
};

static std::vector<winogrande_entry> load_winogrande_from_csv(const std::string& prompt) {
    std::vector<winogrande_entry> result;
    std::istringstream in(prompt);
    std::string line;
    std::array<int, 4> comma_pos;
    while (true) {
        std::getline(in, line);
        if (in.fail() || in.eof()) break;
        int ipos = 0;
        bool quote_open = false;
        for (int i = 0; i < int(line.size()); ++i) {
            if (!quote_open) {
                if (line[i] == ',') {
                    comma_pos[ipos++] = i;
                    if (ipos == 4) break;
                }
                else if (line[i] == '"') {
                    quote_open = true;
                }
            }
            else {
                if (line[i] == '"') {
                    quote_open = false;
                }
            }
        }
        if (ipos != 4) {
            printf("%s: failed to find comma separators in <%s>\n", __func__, line.c_str());
            continue;
        }
        auto sentence = line[comma_pos[0]+1] == '"' ? line.substr(comma_pos[0]+2, comma_pos[1] - comma_pos[0] - 3)
                                                    : line.substr(comma_pos[0]+1, comma_pos[1] - comma_pos[0] - 1);
        auto choice1 = line.substr(comma_pos[1]+1, comma_pos[2] - comma_pos[1] - 1);
        auto choice2 = line.substr(comma_pos[2]+1, comma_pos[3] - comma_pos[2] - 1);
        auto answer  = line.substr(comma_pos[3]+1, line.size() - comma_pos[3] - 1);
        auto index = line.substr(0, comma_pos[0]);
        int where = 0;
        for ( ; where < int(sentence.size()); ++where) {
            if (sentence[where] == '_') break;
        }
        if (where == int(sentence.size())) {
            printf("%s: no _ in <%s>\n", __func__, sentence.c_str());
            continue;
        }
        std::istringstream stream(answer.c_str());
        int i_answer; stream >> i_answer;
        if (stream.fail() || i_answer < 1 || i_answer > 2) {
            printf("%s: failed to parse answer <%s>\n", __func__, answer.c_str());
            continue;
        }
        result.emplace_back();
        auto& wg = result.back();
        wg.first = sentence.substr(0, where);
        wg.second = sentence.substr(where + 1, sentence.size() - where - 1);
        wg.choices[0] = std::move(choice1);
        wg.choices[1] = std::move(choice2);
        wg.answer = i_answer;
    }
    return result;
}

/*
 * Evaluates the Winogrande score.
 * Uses a CSV containing task index, dentence, choice 1, choice 2, answer (1 or 2)
 * You can get one such dataset from e.g. https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp
 * As an example, the 1st row in the above dataset is
 *
 *    0,Sarah was a much better surgeon than Maria so _ always got the easier cases.,Sarah,Maria,2
 *
 */
static void winogrande_score(llama_context * ctx, const gpt_params & params) {

    constexpr int k_min_trailing_ctx = 3;

    auto data = load_winogrande_from_csv(params.prompt);
    if (data.empty()) {
        fprintf(stderr, "%s: no tasks\n", __func__);
        return;
    }

    fprintf(stderr, "%s : loaded %zu tasks from prompt.\n", __func__, data.size());

    if (params.winogrande_tasks > 0 && params.winogrande_tasks < data.size()) {
        fprintf(stderr, "%s : selecting %zu random tasks\n", __func__, params.winogrande_tasks);
        std::mt19937 rng(1);
        std::vector<int> aux(data.size());
        for (int i = 0; i < int(data.size()); ++i) {
            aux[i] = i;
        }
        float scale = 1/(1.f + (float)rng.max());
        std::vector<winogrande_entry> selected;
        selected.resize(params.winogrande_tasks);
        for (int i = 0; i < int(params.winogrande_tasks); ++i) {
            int j = int(scale*rng()*aux.size());
            selected[i] = std::move(data[aux[j]]);
            aux[j] = aux.back();
            aux.pop_back();
        }
        data = std::move(selected);
    }

    fprintf(stderr, "%s : tokenizing selected tasks\n", __func__);

    // This is needed as usual for LLaMA models
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

    for (auto & task : data) {
        task.seq_tokens[0] = ::llama_tokenize(ctx, task.first + task.choices[0] + task.second, add_bos);
        task.seq_tokens[1] = ::llama_tokenize(ctx, task.first + task.choices[1] + task.second, add_bos);

        task.common_prefix = 0;
        for (size_t k = 0; k < task.seq_tokens[0].size(); k++) {
            if (task.seq_tokens[0][k] != task.seq_tokens[1][k]) {
                break;
            }
            task.common_prefix++;
        }

        task.required_tokens = task.common_prefix +
            task.seq_tokens[0].size() - task.common_prefix +
            task.seq_tokens[1].size() - task.common_prefix;

        task.n_base1 = ::llama_tokenize(ctx, task.first + task.choices[0], add_bos).size();
        task.n_base2 = ::llama_tokenize(ctx, task.first + task.choices[1], add_bos).size();
    }

    fprintf(stderr, "%s : calculating winogrande score over selected tasks.\n", __func__);

    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int max_tasks_per_batch = 128;
    const int max_seq = 2*max_tasks_per_batch;

    llama_batch batch = llama_batch_init(n_ctx, 0, max_seq);

    std::vector<float> tok_logits(n_vocab);
    std::vector<float> batch_logits(n_vocab*n_ctx);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

    int n_correct = 0;
    int n_done    = 0;

    for (size_t i0 = 0; i0 < data.size(); i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_batch = 0;

        llama_batch_clear(batch);

        while (n_cur + (int) data[i1].required_tokens <= n_ctx) {
            const int s0 = 2*(i1 - i0);
            if (s0 + 2 > max_seq) {
                break;
            }

            for (size_t i = 0; i < data[i1].common_prefix; ++i) {
                llama_batch_add(batch, data[i1].seq_tokens[0][i], i, { s0 + 0, s0 + 1}, false);
            }
            batch.logits[batch.n_tokens - 1] = true;

            for (int s = 0; s < 2; ++s) {
                for (size_t i = data[i1].common_prefix; i < data[i1].seq_tokens[s].size(); ++i) {
                    llama_batch_add(batch, data[i1].seq_tokens[s][i], i, { s0 + s }, true);
                }
            }

            data[i1].i_batch = i_batch;
            i_batch += data[i1].required_tokens;

            n_cur += data[i1].required_tokens;
            if (++i1 == data.size()) {
                break;
            }
        }

        if (i0 == i1) {
            fprintf(stderr, "%s : task %zu does not fit in the context window\n", __func__, i0);
            return;
        }

        llama_kv_cache_clear(ctx);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            fprintf(stderr, "%s: llama_decode() failed\n", __func__);
            return;
        }

        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            size_t li = n_base1 - 1;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                eval_pairs.push_back(std::make_pair(task.i_batch + li++, task.seq_tokens[0][j+1]));
            }
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            li = task.seq_tokens[0].size() - task.common_prefix + n_base2 - 1;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                eval_pairs.push_back(std::make_pair(task.i_batch + li++, task.seq_tokens[1][j+1]));
            }
        }
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            float score_1st = 0;
            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                score_1st += eval_results[ir++];
            }
            score_1st /= (task.seq_tokens[0].size() - n_base1 - last_1st);

            float score_2nd = 0;
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                score_2nd += eval_results[ir++];
            }
            score_2nd /= (task.seq_tokens[1].size() - n_base2 - last_2nd);

            int result = score_1st > score_2nd ? 1 : 2;

            if (result == task.answer) {
                ++n_correct;
            }
            ++n_done;

            // print the accumulated accuracy mean x 100
            printf("%zu\t%.4lf\t%10.6f  %10.6f  %d  %d\n", i+1, 100.0 * n_correct/n_done, score_1st, score_2nd, result, task.answer);
            fflush(stdout);
        }

        i0 = i1 - 1;
    }

    printf("\n");

    if (n_done < 100) return;

    const float p = 1.f*n_correct/n_done;
    const float sigma = 100.f*sqrt(p*(1-p)/(n_done-1));
    printf("Final Winogrande score(%d tasks): %.4lf +/- %.4lf\n", n_done, 100*p, sigma);
}


int main(int argc, char ** argv) {
    gpt_params params;

    params.n_batch = 512;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    params.logits_all = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.ppl_stride > 0) {
        fprintf(stderr, "Will perform strided perplexity calculation -> adjusting context size from %d to %d\n",
                params.n_ctx, params.n_ctx + params.ppl_stride/2);
        params.n_ctx += params.ppl_stride/2;
    }

    print_build_info();

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", get_system_info(params).c_str());
    }

    struct results_perplexity results;
    if (params.hellaswag) {
        hellaswag_score(ctx, params);
    } else if (params.winogrande) {
        winogrande_score(ctx, params);
    } else {
        results = perplexity(ctx, params);
    }

    llama_print_timings(ctx);
    write_logfile(ctx, params, model, results);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
