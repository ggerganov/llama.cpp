#include <mpi.h>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "common.h"
#include "llama.h"

// Function to print usage
static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);
    if (argc > 0) {
        fprintf(stderr, "\nexample usage:\n");
        fprintf(stderr, "\n    mpirun -np 4 %s -m model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
        fprintf(stderr, "\n");
    }
}

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        if (world_rank == 0) {
            print_usage(argc, argv, params);
        }
        MPI_Finalize();
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "Rank %d: error: unable to load model\n", world_rank);
        MPI_Finalize();
        return 1;
    }

    std::vector<llama_token> tokens_list = ::llama_tokenize(model, params.prompt, true);
    const int n_kv_req = tokens_list.size() + (params.n_predict - tokens_list.size()) * world_size;

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    ctx_params.n_ctx = n_kv_req;
    ctx_params.n_batch = std::max(params.n_predict, world_size);

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "Rank %d: error: failed to create the llama_context\n", world_rank);
        MPI_Finalize();
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);
    if (n_kv_req > n_ctx) {
        fprintf(stderr, "Rank %d: error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n", world_rank, n_kv_req);
        fprintf(stderr, "Rank %d:        either reduce n_parallel or increase n_ctx\n", world_rank);
        MPI_Finalize();
        return 1;
    }

    llama_batch batch = llama_batch_init(std::max(tokens_list.size(), (size_t) world_size), 0, world_size);
    std::vector<llama_seq_id> seq_ids(world_size, 0);
    for (int32_t i = 0; i < world_size; ++i) {
        seq_ids[i] = i;
    }

    for (size_t i = 0; i < tokens_list.size(); ++i) {
        llama_batch_add(batch, tokens_list[i], i, seq_ids, false);
    }

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "Rank %d: failed to eval\n", world_rank);
            MPI_Finalize();
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = llama_token_bos(model);
        }

        llama_batch_clear(batch);
        llama_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
    }

    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Rank %d: llama_decode() failed\n", world_rank);
        MPI_Finalize();
        return 1;
    }

    std::vector<std::string> streams(world_size);
    std::vector<int32_t> i_batch(world_size, batch.n_tokens - 1);

    int n_cur = batch.n_tokens;
    int n_decode = 0;
    const auto t_main_start = ggml_time_us();

    while (n_cur <= params.n_predict) {
        llama_batch_clear(batch);
        for (int32_t i = 0; i < world_size; ++i) {
            if (i_batch[i] < 0) {
                continue;
            }

            auto n_vocab = llama_n_vocab(model);
            auto * logits = llama_get_logits_ith(ctx, i_batch[i]);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
            const int top_k = 40;
            const float top_p = 0.9f;
            const float temp = 0.4f;

            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
            llama_sample_temp(ctx, &candidates_p, temp);

            const llama_token new_token_id = llama_sample_token(ctx, &candidates_p);

            if (llama_token_is_eog(model, new_token_id) || n_cur == params.n_predict) {
                i_batch[i] = -1;
                continue;
            }

            streams[i] += llama_token_to_piece(ctx, new_token_id);
            i_batch[i] = batch.n_tokens;
            llama_batch_add(batch, new_token_id, n_cur, { i }, true);
            n_decode += 1;
        }

        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Rank %d: failed to eval, return code %d\n", world_rank, 1);
            MPI_Finalize();
            return 1;
        }
    }

    std::vector<char> local_output;
    for (const auto & stream : streams) {
        local_output.insert(local_output.end(), stream.begin(), stream.end());
    }

    std::vector<int> recvcounts(world_size);
    int local_size = local_output.size();
    MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(world_size, 0);
    int total_size = 0;
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            displs[i] = total_size;
            total_size += recvcounts[i];
        }
    }

    std::vector<char> global_output(total_size);
    MPI_Gatherv(local_output.data(), local_size, MPI_CHAR, global_output.data(), recvcounts.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::string final_output(global_output.begin(), global_output.end());
        printf("Final output:\n%s\n", final_output.c_str());
    }

    const auto t_main_end = ggml_time_us();
    if (world_rank == 0) {
        printf("Decoded %d tokens in %.2f s, speed: %.2f t/s\n",
               n_decode, (t_main_end - t_main_start) / 1000000.0f,
               n_decode / ((t_main_end - t_main_start) / 1000000.0f));
        llama_print_timings(ctx);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    MPI_Finalize();
    return 0;
}
