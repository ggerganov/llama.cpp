#include "common.h"
#include "llama.h"
#include "build-info.h"

#include <cmath>
#include <ctime>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_exp;
    return probs;
}

void perplexity(llama_context * ctx, const gpt_params & params) {
    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval
    auto tokens = ::llama_tokenize(ctx, params.prompt, true);

    const int n_chunk_max = tokens.size() / params.n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_n_vocab(ctx);
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;

    fprintf(stderr, "%s: calculating perplexity over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * params.n_ctx;
        const int end   = start + params.n_ctx;

        const int num_batches = (params.n_ctx + n_batch - 1) / n_batch;

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (j == 0) {
                tokens[batch_start] = llama_token_bos();
            }

            if (llama_eval(ctx, tokens.data() + batch_start, batch_size, j * n_batch, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return;
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            const auto batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
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
            fprintf(stderr, "%d minutes\n", total_seconds / 60);
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
        for (int j = std::min(512, params.n_ctx / 2); j < params.n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + (j + 0) * n_vocab,
                logits.begin() + (j + 1) * n_vocab);

            const float prob = softmax(tok_logits)[tokens[start + j + 1]];

            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        fflush(stdout);
    }
    printf("\n");
}

void hellaswag_score(llama_context * ctx, const gpt_params & params) {
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

    if( prompt_lines.size() % 6 != 0) {
        fprintf(stderr, "%s : number of lines in prompt not a multiple of 6.\n", __func__);
        return;
    }

    size_t hs_task_count = prompt_lines.size()/6;
    fprintf(stderr, "%s : loaded %zu tasks from prompt.\n", __func__, hs_task_count);

    // This is needed as usual for LLaMA models
    bool prepend_bos = true;

    // Number of tasks to use when computing the score
    if ( params.hellaswag_tasks < hs_task_count  ) {
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
    };

    fprintf(stderr, "%s : selecting %zu %s tasks.\n", __func__, hs_task_count, (randomize_tasks?"randomized":"the first")  );

    // Select and read data from prompt lines
    hs_data_t *hs_data = new hs_data_t[hs_task_count];
    for (size_t i=0; i < hs_task_count; i++) {
        size_t idx = i;

        // Select a random example of those left in the prompt
        if (randomize_tasks) {
            std::uniform_int_distribution<size_t> dist(0, prompt_lines.size()/6-1 ) ;
            idx = dist(rng);
        }

        hs_data[i].context = prompt_lines[idx*6];
        hs_data[i].gold_ending_idx = std::stoi( prompt_lines[idx*6+1] );
        for (size_t j=0; j < 4; j++) {
            hs_data[i].ending[j] = " " + prompt_lines[idx*6+2+j];
        }

        // Delete the selected random example from the prompt
        if (randomize_tasks) {
            prompt_lines.erase( std::next(prompt_lines.begin(),idx*6)  , std::next(prompt_lines.begin(),idx*6+6) );
        }
    }

    fprintf(stderr, "%s : calculating hellaswag score over selected tasks.\n", __func__);
    printf("\ntask\tacc_norm\n");

    double acc = 0.0f;
    const int n_vocab = llama_n_vocab(ctx);

    for (size_t task_idx = 0; task_idx < hs_task_count; task_idx++) {

        // Tokenize the context to count tokens
        std::vector<int> context_embd = ::llama_tokenize(ctx, hs_data[task_idx].context, prepend_bos);
        size_t context_size = context_embd.size();

        for (size_t ending_idx=0;ending_idx<4;ending_idx++) {

            // Tokenize the query
            std::vector<int> query_embd = ::llama_tokenize(ctx, hs_data[task_idx].context + hs_data[task_idx].ending[ending_idx], prepend_bos);
            size_t query_size = query_embd.size();

            // Stop if query wont fit the ctx window
            if (query_size > (size_t)params.n_ctx) {
                fprintf(stderr, "%s : number of tokens in query %zu > n_ctxl\n", __func__, query_size);
                return;
            }

            // Speedup small evaluations by evaluating atleast 32 tokens
            if (query_size < 32) {
                query_embd.resize(32);
            }

            // Evaluate the query
            if (llama_eval(ctx, query_embd.data(), query_embd.size(), 0, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return;
            }

            const auto query_logits = llama_get_logits(ctx);
            std::vector<float> logits;
            logits.insert(logits.end(), query_logits, query_logits + query_size * n_vocab);

            hs_data[task_idx].ending_logprob_count[ending_idx] = 0;
            hs_data[task_idx].ending_logprob[ending_idx] = 0.0f;

            // Calculate the logprobs over the ending
            for (size_t j = context_size-1; j < query_size - 1; j++) {
                // Calculate probability of next token, given the previous ones.
                const std::vector<float> tok_logits(
                    logits.begin() + (j + 0) * n_vocab,
                    logits.begin() + (j + 1) * n_vocab);

                const float prob = softmax(tok_logits)[query_embd[ j + 1]];

                hs_data[task_idx].ending_logprob[ending_idx] += std::log(prob);
                hs_data[task_idx].ending_logprob_count[ending_idx]++;
            }

            // Calculate the mean token logprob for acc_norm
            hs_data[task_idx].ending_logprob[ending_idx] /= hs_data[task_idx].ending_logprob_count[ending_idx];


//            printf("task %lu, ending %lu, whole_len %lu, context_len %lu, ending_logprob_count %lu, ending_logprob %.4f\n",
//                task_idx,ending_idx,whole_size,context_size, hs_data[task_idx].ending_logprob_count[ending_idx], hs_data[task_idx].ending_logprob[ending_idx] );
        }

        // Find the ending with maximum logprob
        size_t ending_logprob_max_idx = -1;
        double ending_logprob_max_val = -INFINITY;
        for (size_t j=0; j < 4; j++) {
            if (hs_data[task_idx].ending_logprob[j] > ending_logprob_max_val) {
                ending_logprob_max_idx = j;
                ending_logprob_max_val =  hs_data[task_idx].ending_logprob[j];
            }
        }

//        printf("max logprob ending idx %lu, gold ending idx %lu\n", ending_logprob_max_idx, hs_data[task_idx].gold_ending_idx);

        // If the gold ending got the maximum logprobe add one accuracy point
        if (ending_logprob_max_idx == hs_data[task_idx].gold_ending_idx) {
            acc += 1.0;
        }

        // Print the accumulated accuracy mean x 100
        printf("%zu\t%.8lf\n",task_idx+1, acc/double(task_idx+1)*100.0);
        fflush(stdout);
    }

    delete [] hs_data;

    printf("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;

    params.n_batch = 512;
    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.perplexity = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model might not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

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

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    if (params.hellaswag) {
        hellaswag_score(ctx, params);
    } else {
        perplexity(ctx, params);
    }

    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
