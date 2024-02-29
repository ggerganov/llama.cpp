#include "common.h"
#include "llama.h"

#include <string>
#include <vector>
#include <format>

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

static void normalize(std::vector<float> in, float* out) {
	float inorm = norm(in);
	for (uint64_t i = 0; i < in.size(); i++) {
		out[i] = in[i] / inorm;
    }
}

static std::vector<std::vector<float>> encode(llama_context* ctx, const std::vector<std::string>& sentences, const std::string& instruction) {
	auto result = std::vector<std::vector<float>>{};

	auto mdl = llama_get_model(ctx);

	for (uint64_t i = 0; i < sentences.size(); i++) {
		auto batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

		// testing with and without EOS - unexpected embeddings in both cases - GritLM seems to have EOS = ""
        std::string input_string = instruction + sentences[i];
        // std::string input_string = sentences[i];
		auto inputs = llama_tokenize(mdl, input_string, true, false);
		// https://github.com/ContextualAI/gritlm/blob/92025b16534712b31b3c4aaaf069350e222bd5f8/gritlm/gritlm.py#L116
		// inputs.push_back(llama_token_eos(mdl));

		// debug tokens - these are matching as referenced in their sample so doesn't appear to be a token issue
		std::for_each(inputs.begin(), inputs.end(), [&ctx](llama_token t) {
            std::printf("[%u:%s]", t, llama_token_to_piece(ctx, t).c_str());
        });
		std::printf("\n");

        // add input to batch (this increments n_tokens)
		for (uint64_t j = 0; j < inputs.size(); j++) {
			llama_batch_add(batch, inputs[j], j, { 0 }, false);
        }

		// clear previous kv_cache values (irrelevant for embeddings)
        llama_kv_cache_clear(ctx);

		// run model
		llama_decode(ctx, batch);

        // get embedding dimensions
        int n_toks = inputs.size();
        int n_embd = llama_n_embd(mdl);

        // allocate embedding output
        std::vector<float> emb_unorm(n_embd, 0.0f);

        // sum up all token embeddings
        for (int k = 0; k < n_toks; k++) {
            float * emb = llama_get_embeddings_ith(ctx, k);
            for (int j = 0; j < n_embd; j++) {
                emb_unorm[j] += emb[j];
            }
        }

        // divide by number of tokens (mean pooling)
        for (int j = 0; j < n_embd; j++) {
            emb_unorm[j] /= n_toks;
        }

		auto emb_norm = std::vector<float>(emb_unorm.size());
		normalize(emb_unorm, emb_norm.data());
		result.push_back(emb_norm);

        // print out emb_norm
        std::printf("embedding %ld: ", i);
        for (int j = 0; j < n_embd; j++) {
            std::printf("%.5f ", emb_norm[j]);
        }
        std::printf("\n");

		llama_batch_free(batch);
	}

	return result;
}

// ./embeddings -m ggml-gritlm-7b-q8_0.gguf -ngl 33
int main(int argc, char* argv[])
{
	gpt_params params;
	if (!gpt_params_parse(argc, argv, params))
		return 1;

	auto mparams = llama_model_params_from_gpt_params(params);
	auto cparams = llama_context_params_from_gpt_params(params);

	mparams.progress_callback = [](std::float_t progress, void* state) {
        std::printf(
            "%s\rLoading model... %u%%\r",
            std::string(32, ' ').c_str(),
            static_cast<std::uint8_t>(progress * 100)
        );
        return true;
    };
	cparams.embedding = true;
    // cparams.do_pooling = false;

	llama_backend_init();

	auto mdl = llama_load_model_from_file(params.model.c_str(), mparams);
	auto ctx = llama_new_context_with_model(mdl, cparams);
	auto bat = llama_batch_init(llama_n_ctx(ctx), 0, 1);

	// ### Embedding/Representation ### taken sample from here:
	// https://github.com/ContextualAI/gritlm?tab=readme-ov-file#basic
	{
		auto instruction = std::string{ "Given a scientific paper title, retrieve the paper's abstract" };

		auto queries = std::vector<std::string>{
            // "hello world",
			"Bitcoin: A Peer-to-Peer Electronic Cash System",
			"Generative Representational Instruction Tuning",
		};

		auto documents = std::vector<std::string>{
			"A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work. The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power. As long as a majority of CPU power is controlled by nodes that are not cooperating to attack the network, they'll generate the longest chain and outpace attackers. The network itself requires minimal structure. Messages are broadcast on a best effort basis, and nodes can leave and rejoin the network at will, accepting the longest proof-of-work chain as proof of what happened while they were gone.",
			"All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8X7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, by no longer requiring separate retrieval and generation models. Models, code, etc. are freely available at https://github.com/ContextualAI/gritlm.",
		};

		auto gritlm_instruction = [](const std::string& instruction) -> std::string {
            return !instruction.empty() ? "<|user|>\n" + instruction + "\n<|embed|>\n" : "<|embed|>\n";
        };

		// No need to add instruction for retrieval documents
		auto d_rep = encode(ctx, documents, gritlm_instruction(""));
		auto q_rep = encode(ctx, queries, gritlm_instruction(instruction));

		auto cosine_sim_q0_d0 = 1 - cosine_similarity(q_rep[0], d_rep[0]);
		auto cosine_sim_q0_d1 = 1 - cosine_similarity(q_rep[0], d_rep[1]);
		auto cosine_sim_q1_d0 = 1 - cosine_similarity(q_rep[1], d_rep[0]);
		auto cosine_sim_q1_d1 = 1 - cosine_similarity(q_rep[1], d_rep[1]);

		std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[0].c_str(), documents[0].c_str(), cosine_sim_q0_d0);
		std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[0].c_str(), documents[1].c_str(), cosine_sim_q0_d1);
		std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[1].c_str(), documents[0].c_str(), cosine_sim_q1_d0);
		std::printf("Cosine similarity between \"%.50s\" and \"%.50s\" is: %.3f\n", queries[1].c_str(), documents[1].c_str(), cosine_sim_q1_d1);
	}

	llama_batch_free(bat);
	llama_free(ctx);
	llama_free_model(mdl);
	llama_backend_free();

	return 0;
}
