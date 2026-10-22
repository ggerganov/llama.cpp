// tree speculative decoding on small random models: tree logits, commits, the greedy walk and draft-simple trees

#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-tree.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

static const int32_t n_vocab  = 256;
static const int32_t n_swa    = 8;
static const int32_t n_prefix = 40;  // longer than the SWA window
static const int32_t n_long   = 300; // longer than the SWA cache

static void set_tensor_data(ggml_tensor * tensor, void * /*userdata*/) {
    std::mt19937 gen(std::hash<std::string>{}(tensor->name));
    std::normal_distribution<float> dis(0.0f, 0.1f);

    std::vector<float> data(ggml_nelements(tensor));
    for (float & v : data) {
        v = dis(gen);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

static gguf_context * make_metadata(const std::string & arch, bool swa) {
    gguf_context * meta = gguf_init_empty();

    const auto key = [&](const char * name) { return arch + "." + name; };

    gguf_set_val_str(meta, "general.architecture", arch.c_str());
    gguf_set_val_u32(meta, key("vocab_size").c_str(),                       n_vocab);
    gguf_set_val_u32(meta, key("context_length").c_str(),                   1024);
    gguf_set_val_u32(meta, key("embedding_length").c_str(),                 64);
    gguf_set_val_u32(meta, key("feed_forward_length").c_str(),              128);
    gguf_set_val_u32(meta, key("block_count").c_str(),                      2);
    gguf_set_val_u32(meta, key("attention.head_count").c_str(),             2);
    gguf_set_val_u32(meta, key("attention.head_count_kv").c_str(),          2);
    gguf_set_val_f32(meta, key("attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    if (swa) {
        // layer 0 uses the sliding window, layer 1 full attention
        gguf_set_val_u32(meta, key("attention.sliding_window").c_str(),         n_swa);
        gguf_set_val_u32(meta, key("attention.sliding_window_pattern").c_str(), 2);
    }

    std::vector<std::string>  tokens(n_vocab);
    std::vector<const char *> token_ptrs(n_vocab);
    std::vector<float>        scores(n_vocab, 0.0f);
    for (int32_t i = 0; i < n_vocab; ++i) {
        tokens[i]     = "tok_" + std::to_string(i);
        token_ptrs[i] = tokens[i].c_str();
    }
    gguf_set_val_str (meta, "tokenizer.ggml.model", "test");
    gguf_set_arr_str (meta, "tokenizer.ggml.tokens", token_ptrs.data(), n_vocab);
    gguf_set_arr_data(meta, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), n_vocab);
    gguf_set_val_u32 (meta, "tokenizer.ggml.eos_token_id", n_vocab - 1);

    return meta;
}

// unified KV cache as the server uses for trees, F32 cells on the CPU
static llama_context * make_ctx(llama_model * model, uint32_t n_seq_max, uint32_t n_ubatch) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 1024;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = n_seq_max;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.kv_unified      = true;
    cparams.swa_full        = false;
    cparams.offload_kqv     = false;
    cparams.op_offload      = false;
    cparams.type_k          = GGML_TYPE_F32;
    cparams.type_v          = GGML_TYPE_F32;

    return llama_init_from_model(model, cparams);
}

static bool decode(llama_context * ctx, common_batch & batch) {
    const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (ret != 0) {
        fprintf(stderr, "%s: llama_process returned %d\n", __func__, ret);
        return false;
    }
    return true;
}

// decode tokens at pos0, pos0 + 1, ... in seq_id, in chunks of n_batch (with outputs, keep the tokens to one chunk)
static bool decode_chain(llama_context * ctx, const llama_tokens & tokens, llama_pos pos0, llama_seq_id seq_id, bool outputs) {
    const size_t n_batch = llama_n_batch(ctx);

    for (size_t i0 = 0; i0 < tokens.size(); i0 += n_batch) {
        common_batch batch(ctx);
        for (size_t i = i0; i < std::min(tokens.size(), i0 + n_batch); ++i) {
            batch.add(tokens[i], pos0 + (llama_pos) i, seq_id, outputs);
        }
        if (!decode(ctx, batch)) {
            return false;
        }
    }
    return true;
}

static std::vector<float> get_logits(llama_context * ctx, int32_t i) {
    const float * logits = llama_get_logits_ith(ctx, i);
    return std::vector<float>(logits, logits + n_vocab);
}

static int32_t argmax(const std::vector<float> & v) {
    return (int32_t) (std::max_element(v.begin(), v.end()) - v.begin());
}

static bool logits_match(const std::vector<float> & a, const std::vector<float> & b, const char * what, int32_t node) {
    float diff = 0.0f;
    for (int32_t t = 0; t < n_vocab; ++t) {
        diff = std::max(diff, std::fabs(a[t] - b[t]));
    }
    if (diff > 1e-4f) {
        fprintf(stderr, "%s: %s, node %d: max logit diff %g\n", __func__, what, node, diff);
        return false;
    }
    return true;
}

static llama_tokens random_tokens(std::mt19937 & rng, int32_t n) {
    // the last id is EOG, keep it out of the drafts
    std::uniform_int_distribution<int32_t> dis(0, n_vocab - 2);
    llama_tokens res(n);
    for (auto & t : res) {
        t = dis(rng);
    }
    return res;
}

// breadth-first tree with up to n_branches leaves, n_nodes nodes and depth_max levels
static common_speculative_tree random_tree(std::mt19937 & rng, int32_t n_branches, int32_t n_nodes, int32_t depth_max) {
    common_speculative_tree tree;

    std::uniform_int_distribution<int32_t> dis_tok(0, n_vocab - 2);
    std::uniform_int_distribution<int32_t> dis_cnt(0, 2);

    std::vector<int32_t> level = { -1 };
    int32_t n_leaves = 1;

    for (int32_t d = 0; d < depth_max && tree.size() < n_nodes && !level.empty(); ++d) {
        std::vector<int32_t> next;
        for (const int32_t u : level) {
            int32_t n_child = 1;
            while (n_child < 3 && n_leaves < n_branches && dis_cnt(rng) > 0) {
                n_child++;
                n_leaves++;
            }

            std::vector<llama_token> used;
            for (int32_t c = 0; c < n_child && tree.size() < n_nodes; ++c) {
                llama_token id;
                do {
                    id = dis_tok(rng);
                } while (std::find(used.begin(), used.end(), id) != used.end());
                used.push_back(id);

                next.push_back(tree.add(id, u));
            }
        }
        level = next;
    }

    tree.finalize();

    return tree;
}

// the nodes from the first child of the root to node
static std::vector<int32_t> path_to(const common_speculative_tree & tree, int32_t node) {
    std::vector<int32_t> path;
    for (int32_t i = node; i >= 0; i = tree.parent[i]) {
        path.push_back(i);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

static llama_tokens path_tokens(const common_speculative_tree & tree, const std::vector<int32_t> & path) {
    llama_tokens res;
    for (const int32_t i : path) {
        res.push_back(tree.tokens[i]);
    }
    return res;
}

static bool check_tree(const common_speculative_tree & tree, int32_t n_branches) {
    bool ok = (tree.empty() ? tree.n_branches() == 0 : tree.n_branches() >= 1) && tree.n_branches() <= n_branches;
    for (int32_t i = 0; i < tree.size(); ++i) {
        ok = ok && tree.parent[i] < i && !tree.branches[i].empty();
        ok = ok && tree.depth[i] == (tree.parent[i] < 0 ? 0 : tree.depth[tree.parent[i]] + 1);
    }
    for (int32_t b = 0; b < tree.n_branches(); ++b) {
        // a branch is the path to its leaf
        for (const int32_t i : path_to(tree, tree.leaves[b])) {
            ok = ok && std::find(tree.branches[i].begin(), tree.branches[i].end(), b) != tree.branches[i].end();
        }
    }
    if (!ok) {
        fprintf(stderr, "%s: invalid tree with %d nodes and %d branches\n", __func__, tree.size(), tree.n_branches());
    }
    return ok;
}

// the other branches hold the same positions and the same number of cells as seqs[0]
static bool branches_are_copies(llama_context * ctx, const std::vector<llama_seq_id> & seqs) {
    auto * mem = llama_get_memory(ctx);

    bool ok = true;
    for (size_t b = 1; b < seqs.size(); ++b) {
        ok = ok && llama_memory_seq_pos_min(mem, seqs[b]) == llama_memory_seq_pos_min(mem, seqs[0]) &&
                   llama_memory_seq_pos_max(mem, seqs[b]) == llama_memory_seq_pos_max(mem, seqs[0]) &&
                   llama_state_seq_get_size(ctx, seqs[b]) == llama_state_seq_get_size(ctx, seqs[0]);
    }
    if (!ok) {
        fprintf(stderr, "%s: the branches are not copies of sequence %d\n", __func__, seqs[0]);
    }
    return ok;
}

// tree for slot 0 next to a chain for slot 1: node logits vs chain decodes, then a random commit vs a fresh decode
static int test_tree_batch(llama_model * model, int32_t n_branches, uint32_t n_ubatch, uint32_t seed) {
    const uint32_t n_slots = 2;

    llama_context * ctx     = make_ctx(model, n_slots*n_branches, n_ubatch);
    llama_context * ctx_ref = make_ctx(model, 1, 64);
    if (!ctx || !ctx_ref) {
        fprintf(stderr, "%s: failed to create the contexts\n", __func__);
        return 1;
    }

    int n_fail = 0;

    std::mt19937 rng(seed);

    common_memory mem;
    mem.init(ctx);

    std::vector<llama_seq_id> seqs;
    for (int32_t b = 0; b < n_branches; ++b) {
        seqs.push_back(common_speculative_branch_seq(n_slots, n_branches, 0, b));
    }

    const llama_tokens prefix0 = random_tokens(rng, n_prefix);
    const llama_tokens prefix1 = random_tokens(rng, n_prefix/2);
    const llama_tokens chain1  = random_tokens(rng, 3);
    const llama_token  root    = random_tokens(rng, 1)[0];

    const common_speculative_tree tree = random_tree(rng, n_branches, 12, 5);
    n_fail += !check_tree(tree, n_branches);

    const llama_pos pos0 = n_prefix;

    if (!decode_chain(ctx, prefix0, 0, 0, false) || !decode_chain(ctx, prefix1, 0, 1, false)) {
        return n_fail + 1;
    }

    common_speculative_tree_fork(mem, seqs);

    common_batch batch(ctx);
    batch.add(root, pos0, tree.node_seqs(-1, seqs), true);
    for (int32_t i = 0; i < tree.size(); ++i) {
        batch.add(tree.tokens[i], pos0 + 1 + tree.depth[i], tree.node_seqs(i, seqs), true);
    }
    for (size_t k = 0; k < chain1.size(); ++k) {
        batch.add(chain1[k], (llama_pos) (prefix1.size() + k), 1, true);
    }
    if (!decode(ctx, batch)) {
        return n_fail + 1;
    }

    // every branch decoded as a chain
    for (int32_t b = 0; b < tree.n_branches(); ++b) {
        const auto path = path_to(tree, tree.leaves[b]);

        llama_tokens tokens = { root };
        for (const llama_token t : path_tokens(tree, path)) {
            tokens.push_back(t);
        }

        llama_memory_clear(llama_get_memory(ctx_ref), true);
        if (!decode_chain(ctx_ref, prefix0, 0, 0, false) || !decode_chain(ctx_ref, tokens, pos0, 0, true)) {
            return n_fail + 1;
        }

        n_fail += !logits_match(get_logits(ctx, 0), get_logits(ctx_ref, 0), "root", -1);
        for (size_t d = 0; d < path.size(); ++d) {
            n_fail += !logits_match(get_logits(ctx, 1 + path[d]), get_logits(ctx_ref, (int32_t) d + 1), "node", path[d]);
        }
    }

    // the other slot in the same batch
    {
        llama_memory_clear(llama_get_memory(ctx_ref), true);
        if (!decode_chain(ctx_ref, prefix1, 0, 0, false) || !decode_chain(ctx_ref, chain1, (llama_pos) prefix1.size(), 0, true)) {
            return n_fail + 1;
        }
        for (size_t k = 0; k < chain1.size(); ++k) {
            n_fail += !logits_match(get_logits(ctx, 1 + tree.size() + (int32_t) k), get_logits(ctx_ref, (int32_t) k), "slot 1", (int32_t) k);
        }
    }

    // commit the path to a random node, -1 = only the root
    const int32_t node = std::uniform_int_distribution<int32_t>(-1, tree.size() - 1)(rng);
    const auto path = path_to(tree, node);

    common_speculative_tree_commit(mem, tree, path, seqs, pos0);

    auto * mem_ctx = llama_get_memory(ctx);
    n_fail += !branches_are_copies(ctx, seqs);
    if (llama_memory_seq_pos_min(mem_ctx, 0) != 0 || llama_memory_seq_pos_max(mem_ctx, 0) != pos0 + (llama_pos) path.size()) {
        fprintf(stderr, "%s: slot 0 has positions [%d, %d], expected [0, %d]\n", __func__,
                llama_memory_seq_pos_min(mem_ctx, 0), llama_memory_seq_pos_max(mem_ctx, 0), pos0 + (int) path.size());
        n_fail++;
    }
    if (llama_memory_seq_pos_max(mem_ctx, 1) != (llama_pos) (prefix1.size() + chain1.size()) - 1) {
        fprintf(stderr, "%s: slot 1 changed by the commit\n", __func__);
        n_fail++;
    }

    // the next token sees the prefix, the root and the path only
    const llama_token next = random_tokens(rng, 1)[0];
    if (!decode_chain(ctx, { next }, pos0 + 1 + (llama_pos) path.size(), 0, true)) {
        return n_fail + 1;
    }

    llama_tokens tokens = prefix0;
    tokens.push_back(root);
    for (const llama_token t : path_tokens(tree, path)) {
        tokens.push_back(t);
    }
    tokens.push_back(next);

    llama_memory_clear(llama_get_memory(ctx_ref), true);
    if (!decode_chain(ctx_ref, tokens, 0, 0, true)) {
        return n_fail + 1;
    }
    n_fail += !logits_match(get_logits(ctx, 0), get_logits(ctx_ref, (int32_t) tokens.size() - 1), "next", node);

    // same cells in the slot as in the decode from scratch
    if (llama_state_seq_get_size(ctx, 0) != llama_state_seq_get_size(ctx_ref, 0)) {
        fprintf(stderr, "%s: slot 0 state size %zu, expected %zu\n", __func__,
                llama_state_seq_get_size(ctx, 0), llama_state_seq_get_size(ctx_ref, 0));
        n_fail++;
    }

    llama_free(ctx_ref);
    llama_free(ctx);

    return n_fail;
}

// rounds of tree + commit on a long prefix: the branches stay copies of the slot, out-of-window SWA cells get reused
static int test_tree_rounds(llama_model * model, int32_t n_branches, uint32_t seed) {
    llama_context * ctx     = make_ctx(model, n_branches, 64);
    llama_context * ctx_ref = make_ctx(model, 1, 64);
    if (!ctx || !ctx_ref) {
        fprintf(stderr, "%s: failed to create the contexts\n", __func__);
        return 1;
    }

    int n_fail = 0;

    std::mt19937 rng(seed);

    common_memory mem;
    mem.init(ctx);

    std::vector<llama_seq_id> seqs;
    for (int32_t b = 0; b < n_branches; ++b) {
        seqs.push_back(common_speculative_branch_seq(1, n_branches, 0, b));
    }

    llama_tokens history = random_tokens(rng, n_long);
    if (!decode_chain(ctx, history, 0, 0, false)) {
        return n_fail + 1;
    }

    common_speculative_tree_fork(mem, seqs);

    // enough rounds to fill the SWA cache with cells shared by all branches
    for (int round = 0; round < 40; ++round) {
        n_fail += !common_speculative_tree_synced(mem, seqs) || !branches_are_copies(ctx, seqs);

        const llama_token root = random_tokens(rng, 1)[0];
        const llama_pos   pos0 = (llama_pos) history.size();

        const common_speculative_tree tree = random_tree(rng, n_branches, 12, 5);

        common_batch batch(ctx);
        batch.add(root, pos0, tree.node_seqs(-1, seqs), true);
        for (int32_t i = 0; i < tree.size(); ++i) {
            batch.add(tree.tokens[i], pos0 + 1 + tree.depth[i], tree.node_seqs(i, seqs), true);
        }
        if (!decode(ctx, batch)) {
            return n_fail + 1;
        }

        for (int32_t b = 0; b < tree.n_branches(); ++b) {
            const auto path = path_to(tree, tree.leaves[b]);

            llama_tokens tokens = { root };
            for (const llama_token t : path_tokens(tree, path)) {
                tokens.push_back(t);
            }

            llama_memory_clear(llama_get_memory(ctx_ref), true);
            if (!decode_chain(ctx_ref, history, 0, 0, false) || !decode_chain(ctx_ref, tokens, pos0, 0, true)) {
                return n_fail + 1;
            }

            n_fail += !logits_match(get_logits(ctx, 0), get_logits(ctx_ref, 0), "root", -1);
            for (size_t d = 0; d < path.size(); ++d) {
                n_fail += !logits_match(get_logits(ctx, 1 + path[d]), get_logits(ctx_ref, (int32_t) d + 1), "node", path[d]);
            }
        }

        const int32_t node = std::uniform_int_distribution<int32_t>(-1, tree.size() - 1)(rng);
        const auto path = path_to(tree, node);

        common_speculative_tree_commit(mem, tree, path, seqs, pos0);

        history.push_back(root);
        for (const llama_token t : path_tokens(tree, path)) {
            history.push_back(t);
        }
    }

    n_fail += !common_speculative_tree_synced(mem, seqs) || !branches_are_copies(ctx, seqs);

    // the next token sees exactly the committed history
    const llama_token next = random_tokens(rng, 1)[0];
    if (!decode_chain(ctx, { next }, (llama_pos) history.size(), 0, true)) {
        return n_fail + 1;
    }

    llama_memory_clear(llama_get_memory(ctx_ref), true);
    if (!decode_chain(ctx_ref, history, 0, 0, false) || !decode_chain(ctx_ref, { next }, (llama_pos) history.size(), 0, true)) {
        return n_fail + 1;
    }
    n_fail += !logits_match(get_logits(ctx, 0), get_logits(ctx_ref, 0), "next", -1);

    llama_free(ctx_ref);
    llama_free(ctx);

    if (n_fail > 0) {
        fprintf(stderr, "%s: n_branches = %d, seed = %u failed\n", __func__, n_branches, seed);
    }

    return n_fail;
}

// greedy continuation of prefix + root, n tokens
static llama_tokens greedy(llama_context * ctx, const llama_tokens & prefix, llama_token root, int32_t n) {
    llama_memory_clear(llama_get_memory(ctx), true);
    decode_chain(ctx, prefix, 0, 0, false);

    llama_tokens res;
    llama_token cur = root;
    for (int32_t i = 0; i < n; ++i) {
        decode_chain(ctx, { cur }, (llama_pos) (prefix.size() + i), 0, true);
        cur = argmax(get_logits(ctx, 0));
        res.push_back(cur);
    }
    return res;
}

static common_sampler * make_sampler(llama_model * model, float temp) {
    common_params_sampling sparams;
    sparams.temp  = temp;
    sparams.top_k = 20;
    sparams.seed  = 1234;
    return common_sampler_init(model, sparams);
}

// decode root + tree in the branch sequences, returns the batch indices
static std::vector<int32_t> decode_tree(llama_context * ctx, const common_memory & mem, const common_speculative_tree & tree,
        const std::vector<llama_seq_id> & seqs, llama_token root, llama_pos pos0) {
    common_speculative_tree_fork(mem, seqs);

    common_batch batch(ctx);
    std::vector<int32_t> idxs;
    idxs.push_back(batch.add(root, pos0, tree.node_seqs(-1, seqs), true));
    for (int32_t i = 0; i < tree.size(); ++i) {
        idxs.push_back(batch.add(tree.tokens[i], pos0 + 1 + tree.depth[i], tree.node_seqs(i, seqs), true));
    }
    if (!decode(ctx, batch)) {
        return {};
    }
    return idxs;
}

static int test_tree_accept(llama_model * model, uint32_t seed) {
    const int32_t n_branches = 3;
    const int32_t n_draft    = 6;

    llama_context * ctx     = make_ctx(model, n_branches, 64);
    llama_context * ctx_ref = make_ctx(model, 1, 64);
    if (!ctx || !ctx_ref) {
        fprintf(stderr, "%s: failed to create the contexts\n", __func__);
        return 1;
    }

    int n_fail = 0;

    std::mt19937 rng(seed);

    common_memory mem;
    mem.init(ctx);

    const std::vector<llama_seq_id> seqs = { 0, 1, 2 };

    const llama_tokens prefix = random_tokens(rng, n_prefix);
    const llama_token  root   = random_tokens(rng, 1)[0];
    const llama_pos    pos0   = n_prefix;

    const llama_tokens target = greedy(ctx_ref, prefix, root, n_draft + 1);

    if (!decode_chain(ctx, prefix, 0, 0, false)) {
        return n_fail + 1;
    }

    // a chain that leaves the greedy path at position m (m == n_draft: never), at temp 0 and with sampling
    for (int32_t m = 0; m <= n_draft; m += 2) {
        llama_tokens draft(target.begin(), target.begin() + n_draft);
        if (m < n_draft) {
            draft[m] = (draft[m] + 1) % (n_vocab - 1);
        }

        common_speculative_tree tree;
        tree.from_chain(draft);

        const auto idxs = decode_tree(ctx, mem, tree, seqs, root, pos0);
        if (idxs.empty()) {
            return n_fail + 1;
        }

        for (const float temp : { 0.0f, 0.8f }) {
            common_sampler * smpl_chain = make_sampler(model, temp);
            common_sampler * smpl_tree  = make_sampler(model, temp);

            std::vector<int32_t> path;
            const auto res_chain = common_sampler_sample_and_accept_n(smpl_chain, ctx, idxs, draft);
            const auto res_tree  = common_speculative_tree_sample_and_accept(smpl_tree, ctx, tree, idxs, path);

            if (res_chain != res_tree || path.size() + 1 != res_tree.size()) {
                fprintf(stderr, "%s: m = %d, temp = %.1f: the tree walk accepted %zu tokens, the chain %zu\n",
                        __func__, m, temp, res_tree.size(), res_chain.size());
                n_fail++;
            }
            if (temp == 0.0f && res_tree.size() != (size_t) std::min(m, n_draft) + 1) {
                fprintf(stderr, "%s: m = %d: %zu tokens accepted at temp 0\n", __func__, m, res_tree.size());
                n_fail++;
            }

            common_sampler_free(smpl_chain);
            common_sampler_free(smpl_tree);
        }

        llama_memory_seq_rm(llama_get_memory(ctx), 0, pos0, -1);
    }

    // branch 0 leaves the greedy path at once, branch 1 follows it, so the walk has to take branch 1
    {
        common_speculative_tree tree;
        int32_t a = -1;
        int32_t b = -1;
        for (int32_t d = 0; d < n_draft/2; ++d) {
            a = tree.add(d == 0 ? (target[0] + 1) % (n_vocab - 1) : random_tokens(rng, 1)[0], a);
            b = tree.add(target[d], b);
        }
        tree.finalize();
        n_fail += !check_tree(tree, n_branches);

        const auto idxs = decode_tree(ctx, mem, tree, seqs, root, pos0);
        if (idxs.empty()) {
            return n_fail + 1;
        }

        common_sampler * smpl = make_sampler(model, 0.0f);

        std::vector<int32_t> path;
        const auto res = common_speculative_tree_sample_and_accept(smpl, ctx, tree, idxs, path);

        const llama_tokens expected(target.begin(), target.begin() + n_draft/2 + 1);
        if (res != expected || tree.branches[path.back()][0] != 1) {
            fprintf(stderr, "%s: the walk did not follow the greedy branch\n", __func__);
            n_fail++;
        }

        common_sampler_free(smpl);

        common_speculative_tree_commit(mem, tree, path, seqs, pos0);

        // the next token after the accepted path
        if (!decode_chain(ctx, { res.back() }, pos0 + 1 + (llama_pos) path.size(), 0, true)) {
            return n_fail + 1;
        }
        greedy(ctx_ref, prefix, root, (int32_t) res.size() + 1);
        n_fail += !logits_match(get_logits(ctx, 0), get_logits(ctx_ref, 0), "after the commit", -1);
    }

    llama_free(ctx_ref);
    llama_free(ctx);

    return n_fail;
}

// draft-simple with the target model as the draft
static int test_draft_simple(llama_model * model, int32_t n_branches, float p_min, uint32_t seed) {
    const int32_t n_draft = 8;

    llama_context * ctx_tgt = make_ctx(model, n_branches, 64);
    llama_context * ctx_dft = make_ctx(model, n_branches, 64);
    llama_context * ctx_ref = make_ctx(model, 1, 64);
    if (!ctx_tgt || !ctx_dft || !ctx_ref) {
        fprintf(stderr, "%s: failed to create the contexts\n", __func__);
        return 1;
    }

    int n_fail = 0;

    std::mt19937 rng(seed);

    common_params_speculative params;
    params.types            = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    params.draft.n_max      = n_draft;
    params.draft.n_branches = n_branches;
    params.draft.p_split    = 0.05f;
    params.draft.p_min      = p_min;
    params.draft.ctx_tgt    = ctx_tgt;
    params.draft.ctx_dft    = ctx_dft;

    common_speculative * spec = common_speculative_init(params, 1);
    if (!spec) {
        fprintf(stderr, "%s: failed to init the speculative context\n", __func__);
        return 1;
    }

    const llama_tokens prefix = random_tokens(rng, n_prefix);
    const llama_token  root   = random_tokens(rng, 1)[0];
    const llama_pos    pos0   = n_prefix;

    if (!decode_chain(ctx_dft, prefix, 0, 0, false)) {
        return n_fail + 1;
    }

    for (const int32_t n_max : { n_draft, 5 }) {
        llama_tokens result;
        common_speculative_tree tree;

        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp = {
            /* .drafting = */ true,
            /* .n_max    = */ n_max,
            /* .pos0     = */ pos0,
            /* .id_last  = */ root,
            /* .prompt   = */ &prefix,
            /* .result   = */ &result,
        };
        dp.result_tree = &tree;

        common_speculative_draft(spec);

        n_fail += !check_tree(tree, n_branches);

        if (tree.tokens != result) {
            fprintf(stderr, "%s: the draft does not hold the tokens of the tree\n", __func__);
            n_fail++;
        }
        if (p_min == 0.0f && tree.size() != n_max) {
            fprintf(stderr, "%s: the tree has %d nodes, expected %d\n", __func__, tree.size(), n_max);
            n_fail++;
        }

        // the slot sequence is truncated by the caller, the branches the drafter used are left as copies of it
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos0, -1);
        for (int32_t b = 1; b < n_branches; ++b) {
            const llama_seq_id seq = common_speculative_branch_seq(1, n_branches, 0, b);
            if (llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq) != -1) {
                n_fail += !branches_are_copies(ctx_dft, { 0, seq });
            }
        }

        // the first child of a node is the top candidate of the draft after its path
        for (int32_t node = -1; node < tree.size(); ++node) {
            const int32_t c = node < 0 ? 0 : (int32_t) (std::find(tree.parent.begin(), tree.parent.end(), node) - tree.parent.begin());
            if (c >= tree.size()) {
                continue;
            }

            llama_tokens tokens = { root };
            for (const llama_token t : path_tokens(tree, path_to(tree, node))) {
                tokens.push_back(t);
            }

            llama_memory_clear(llama_get_memory(ctx_ref), true);
            if (!decode_chain(ctx_ref, prefix, 0, 0, false) || !decode_chain(ctx_ref, tokens, pos0, 0, true)) {
                return n_fail + 1;
            }
            if (tree.tokens[c] != argmax(get_logits(ctx_ref, (int32_t) tokens.size() - 1))) {
                fprintf(stderr, "%s: node %d is not the top candidate after node %d\n", __func__, c, node);
                n_fail++;
            }
        }
    }

    common_speculative_free(spec);

    llama_free(ctx_ref);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);

    return n_fail;
}

int main(void) {
    llama_backend_init();

    int n_fail = 0;

    for (const bool swa : { false, true }) {
        gguf_context * meta = make_metadata(swa ? "gemma3" : "llama", swa);

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;

        llama_model * model = llama_model_init_from_user(meta, set_tensor_data, nullptr, mparams);
        if (model == nullptr) {
            fprintf(stderr, "%s: failed to create the model\n", __func__);
            return 1;
        }

        for (const int32_t n_branches : { 2, 3, 4 }) {
            for (const uint32_t n_ubatch : { 2u, 64u }) {
                for (uint32_t seed = 0; seed < 4; ++seed) {
                    const int n = test_tree_batch(model, n_branches, n_ubatch, seed);
                    if (n > 0) {
                        fprintf(stderr, "%s: test_tree_batch(swa = %d, n_branches = %d, n_ubatch = %u, seed = %u) failed\n",
                                __func__, swa, n_branches, n_ubatch, seed);
                    }
                    n_fail += n;
                }
            }
        }

        for (const int32_t n_branches : { 2, 4 }) {
            for (uint32_t seed = 0; seed < 2; ++seed) {
                n_fail += test_tree_rounds(model, n_branches, seed);
            }
        }

        for (uint32_t seed = 0; seed < 4; ++seed) {
            n_fail += test_tree_accept(model, seed);
        }

        for (const int32_t n_branches : { 2, 4 }) {
            for (const float p_min : { 0.0f, 0.1f }) {
                n_fail += test_draft_simple(model, n_branches, p_min, 7 + n_branches);
            }
        }

        llama_model_free(model);
        gguf_free(meta);
    }

    llama_backend_free();

    printf("%s: %s\n", __func__, n_fail == 0 ? "OK" : "FAILED");

    return n_fail == 0 ? 0 : 1;
}
