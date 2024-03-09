#include "common.h"
#include "llama.h"

#include <string>
#include <vector>

static float dot_product(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot = 0.0f;
    for (uint64_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
    }
    return dot;
}

static float norm(const std::vector<float>& v) {
    return std::sqrt(dot_product(v, v));
}

static float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    return dot_product(v1, v2) / (norm(v1) * norm(v2));
}

static void normalize(const std::vector<float>& in, float* out) {
    float inorm = norm(in);
    for (uint64_t i = 0; i < in.size(); i++) {
        out[i] = in[i] / inorm;
    }
}

static std::vector<std::vector<float>> encode(llama_context* ctx, const std::vector<std::string>& sentences, const std::string& instruction) {
    auto result = std::vector<std::vector<float>>{};

    auto mdl = llama_get_model(ctx);
    auto batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

    for (uint64_t i = 0; i < sentences.size(); i++) {
        llama_batch_clear(batch);

        std::string input_string = instruction + sentences[i];
        std::vector<llama_token> inputs = llama_tokenize(mdl, input_string, true, false);
        auto n_toks = (int32_t)inputs.size();

        // testing with and without EOS - unexpected embeddings in both cases - GritLM seems to have EOS = ""
        // https://github.com/ContextualAI/gritlm/blob/92025b16534712b31b3c4aaaf069350e222bd5f8/gritlm/gritlm.py#L116
        // inputs.push_back(llama_token_eos(mdl));

        // we want to ignore instruction tokens for mean pooling
        std::vector<llama_token> inputs_instruct = llama_tokenize(mdl, instruction, true, false);
        auto n_inst = (int32_t)inputs_instruct.size();

        /*
        // debug tokens - should be matching as referenced in the GritLM sample
        std::for_each(inputs.begin(), inputs.end(), [&ctx](llama_token t) {
            std::printf("[%u:%s]", t, llama_token_to_piece(ctx, t).c_str());
        });
        std::printf("\n");
        */

        // add input to batch (this increments n_tokens)
        for (int32_t j = 0; j < n_toks; j++) {
            llama_batch_add(batch, inputs[j], j, { 0 }, j >= n_inst);
        }

        // clear previous kv_cache values (irrelevant for embeddings)
        llama_kv_cache_clear(ctx);

        // run model
        llama_decode(ctx, batch);

        // get embedding dimensions
        uint64_t n_embd = llama_n_embd(mdl);

        // allocate embedding output
        std::vector<float> emb_unorm(n_embd, 0.0f);

        // sum up all token embeddings
        for (int32_t k = n_inst; k < n_toks; k++) {
            float* emb = llama_get_embeddings_ith(ctx, k);
            for (uint64_t j = 0; j < n_embd; j++) {
                emb_unorm[j] += emb[j];
            }
        }

        // divide by number of tokens (mean pooling)
        uint64_t n_sent = n_toks - n_inst;
        for (uint64_t j = 0; j < n_embd; j++) {
            emb_unorm[j] /= n_sent;
        }

        auto emb_norm = std::vector<float>(emb_unorm.size());
        normalize(emb_unorm, emb_norm.data());
        result.push_back(emb_norm);

        /*
        // print out emb_norm
        std::printf("embedding %ld: ", i);
        for (uint64_t j = 0; j < n_embd; j++) {
            std::printf("%.5f ", emb_norm[j]);
        }
        std::printf("\n\n");
        */
    }

    llama_batch_free(batch);
    return result;
}

static std::string aggregate_pieces(const std::vector<std::string>& pieces) {
    // calculate total length required
    size_t length = 0;
    for (const auto& str : pieces) {
        length += str.size();
    }

    // reserve memory
    std::string result;
    result.reserve(length);

    // append pieces
    for (const auto& str : pieces) {
        result += str;
    }

    return result;
}

static std::string generate(llama_context* ctx, const std::string& prompt, bool stream) {
    std::vector<std::string> pieces;

    const llama_model* mdl = llama_get_model(ctx);
    llama_batch bat = llama_batch_init(llama_n_batch(ctx), 0, 1);

    std::vector<llama_token> inputs = llama_tokenize(mdl, prompt, false, true);
    int32_t i_current_token = 0;

    while (true) {
        llama_batch_clear(bat);

        for (int32_t i = 0; i < inputs.size(); i++)
            llama_batch_add(bat, inputs[i], i_current_token++, { 0 }, i == inputs.size() - 1);

        inputs.clear();

        llama_decode(ctx, bat);

        auto logits = llama_get_logits_ith(ctx, bat.n_tokens - 1);

        auto candidates = std::vector<llama_token_data>(llama_n_vocab(mdl));
        for (int32_t token = 0; token < candidates.size(); token++)
            candidates[token] = llama_token_data{ token, logits[token], 0.0f };

        auto candidates_p = llama_token_data_array{ candidates.data(), candidates.size(), false };

        llama_token token = llama_sample_token_greedy(ctx, &candidates_p);
        if (token == llama_token_eos(mdl))
            break;

        std::string piece = llama_token_to_piece(ctx, token);
        if (stream) {
            std::printf("%s", piece.c_str());
            std::fflush(stdout);
        }

        pieces.push_back(piece);
        inputs.push_back(token);
    }

    if (stream) {
        std::printf("\n");
    }

    llama_batch_free(bat);

    return aggregate_pieces(pieces);
}

static std::string gritlm_instruction(const std::string& instruction) {
    return !instruction.empty() ? "<|user|>\n" + instruction + "\n<|embed|>\n" : "<|embed|>\n";
}

int main(int argc, char* argv[])
{
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    llama_model_params mparams = llama_model_params_from_gpt_params(params);
    llama_context_params cparams = llama_context_params_from_gpt_params(params);

    llama_backend_init();

    llama_model* mdl = llama_load_model_from_file(params.model.c_str(), mparams);

    // create new context - set to embedding mode
    llama_context* embd_ctx = llama_new_context_with_model(mdl, cparams);
    llama_set_embeddings(embd_ctx, true);

    // create new context - default mode is causal
    llama_context* causal_ctx = llama_new_context_with_model(mdl, cparams);

    // ### Embedding/Representation ### samples taken from here:
    // https://github.com/ContextualAI/gritlm?tab=readme-ov-file#basic
    {
        std::string instruction = "Given a scientific paper title, retrieve the paper's abstract";

        std::vector<std::string> queries = {
            "Bitcoin: A Peer-to-Peer Electronic Cash System",
            "Generative Representational Instruction Tuning",
        };

        std::vector<std::string> documents = {
            "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work. The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power. As long as a majority of CPU power is controlled by nodes that are not cooperating to attack the network, they'll generate the longest chain and outpace attackers. The network itself requires minimal structure. Messages are broadcast on a best effort basis, and nodes can leave and rejoin the network at will, accepting the longest proof-of-work chain as proof of what happened while they were gone.",
            "All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8X7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, by no longer requiring separate retrieval and generation models. Models, code, etc. are freely available at https://github.com/ContextualAI/gritlm.",
        };

        // No need to add instruction for retrieval documents
        std::vector<std::vector<float>> d_rep = encode(embd_ctx, documents, gritlm_instruction(""));
        std::vector<std::vector<float>> q_rep = encode(embd_ctx, queries, gritlm_instruction(instruction));

        float cosine_sim_q0_d0 = cosine_similarity(q_rep[0], d_rep[0]);
        float cosine_sim_q0_d1 = cosine_similarity(q_rep[0], d_rep[1]);
        float cosine_sim_q1_d0 = cosine_similarity(q_rep[1], d_rep[0]);
        float cosine_sim_q1_d1 = cosine_similarity(q_rep[1], d_rep[1]);

        std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[0].c_str(), documents[0].c_str(), cosine_sim_q0_d0);
        std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[0].c_str(), documents[1].c_str(), cosine_sim_q0_d1);
        std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[1].c_str(), documents[0].c_str(), cosine_sim_q1_d0);
        std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[1].c_str(), documents[1].c_str(), cosine_sim_q1_d1);
    }

    // ### Generation ###
    // # GritLM models are not finetuned with system prompts, as you can just include system-like instructions together with your user instruction
    {
        const std::string prompt = "<|user|>\nPlease write me a poem about my recent hike of Mt. Fuji at midnight in the style of Shakespeare.\n<|assistant|>\n";
        std::string response = generate(causal_ctx, prompt, true);
    }

    llama_free(embd_ctx);
    llama_free(causal_ctx);

    llama_free_model(mdl);
    llama_backend_free();

    return 0;
}
