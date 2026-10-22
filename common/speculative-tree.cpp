#include "speculative-tree.h"

#include "sampling.h"

void common_speculative_tree::clear() {
    tokens.clear();
    parent.clear();
    depth.clear();
    leaves.clear();
    branches.clear();
}

int32_t common_speculative_tree::add(llama_token id, int32_t i_parent) {
    GGML_ASSERT(i_parent >= -1 && i_parent < size());

    tokens.push_back(id);
    parent.push_back(i_parent);

    return size() - 1;
}

void common_speculative_tree::from_chain(const llama_tokens & draft) {
    clear();

    for (size_t i = 0; i < draft.size(); ++i) {
        add(draft[i], (int32_t) i - 1);
    }

    finalize();
}

void common_speculative_tree::truncate(int32_t n) {
    if (n >= size()) {
        return;
    }

    tokens.resize(n);
    parent.resize(n);

    finalize();
}

void common_speculative_tree::finalize() {
    const int32_t n = size();

    GGML_ASSERT((int32_t) parent.size() == n);

    std::vector<int32_t> first_child(n, -1);

    depth.resize(n);
    for (int32_t i = 0; i < n; ++i) {
        GGML_ASSERT(parent[i] >= -1 && parent[i] < i);

        if (parent[i] < 0) {
            depth[i] = 0;
        } else {
            depth[i] = depth[parent[i]] + 1;
            if (first_child[parent[i]] < 0) {
                first_child[parent[i]] = i;
            }
        }
    }

    leaves.clear();
    if (n > 0) {
        // node 0 is the first child of the root
        int32_t cur = 0;
        while (first_child[cur] >= 0) {
            cur = first_child[cur];
        }
        leaves.push_back(cur);

        for (int32_t i = 0; i < n; ++i) {
            if (first_child[i] < 0 && i != leaves[0]) {
                leaves.push_back(i);
            }
        }
    }

    branches.assign(n, {});
    for (int32_t b = 0; b < n_branches(); ++b) {
        for (int32_t i = leaves[b]; i >= 0; i = parent[i]) {
            branches[i].push_back(b);
        }
    }
}

int32_t common_speculative_tree::child(int32_t node, llama_token id) const {
    for (int32_t i = node + 1; i < size(); ++i) {
        if (parent[i] == node && tokens[i] == id) {
            return i;
        }
    }

    return -1;
}

std::vector<llama_seq_id> common_speculative_tree::node_seqs(int32_t node, const std::vector<llama_seq_id> & seqs) const {
    GGML_ASSERT((int32_t) seqs.size() >= n_branches());

    if (node < 0) {
        return std::vector<llama_seq_id>(seqs.begin(), seqs.begin() + n_branches());
    }

    std::vector<llama_seq_id> res;
    res.reserve(branches[node].size());
    for (const int32_t b : branches[node]) {
        res.push_back(seqs[b]);
    }

    return res;
}

llama_seq_id common_speculative_branch_seq(uint32_t n_seq, int32_t n_branches, llama_seq_id seq_id, int32_t b) {
    GGML_ASSERT(b >= 0 && b < n_branches);

    return b == 0 ? seq_id : (llama_seq_id) n_seq + seq_id*(n_branches - 1) + (b - 1);
}

void common_speculative_tree_fork(const common_memory & mem, const std::vector<llama_seq_id> & seqs) {
    for (size_t b = 1; b < seqs.size(); ++b) {
        mem.seq_rm(seqs[b], -1, -1);
        mem.seq_cp(seqs[0], seqs[b], -1, -1);
    }
}

bool common_speculative_tree_synced(const common_memory & mem, const std::vector<llama_seq_id> & seqs) {
    for (llama_context * ctx : { mem.ctx_tgt, mem.ctx_dft }) {
        if (ctx == nullptr) {
            continue;
        }

        auto * m = llama_get_memory(ctx);

        for (size_t b = 1; b < seqs.size(); ++b) {
            if (llama_memory_seq_pos_min(m, seqs[b]) != llama_memory_seq_pos_min(m, seqs[0]) ||
                llama_memory_seq_pos_max(m, seqs[b]) != llama_memory_seq_pos_max(m, seqs[0])) {
                return false;
            }
        }
    }

    return true;
}

std::vector<llama_token> common_speculative_tree_sample_and_accept(
        common_sampler * smpl,
        llama_context * ctx,
        const common_speculative_tree & tree,
        const std::vector<int32_t> & idxs,
        std::vector<int32_t> & path) {
    GGML_ASSERT((int32_t) idxs.size() == tree.size() + 1 && "idxs.size() must be tree.size() + 1");

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::vector<llama_token> result;

    path.clear();

    int32_t cur = -1;
    while (true) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[cur + 1]);

        common_sampler_accept(smpl, id, true);

        result.push_back(id);

        // draft tokens after an EOG are not output, do not keep them in the context
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        cur = tree.child(cur, id);
        if (cur < 0) {
            break;
        }

        path.push_back(cur);
    }

    return result;
}

void common_speculative_tree_commit(
        const common_memory & mem,
        const common_speculative_tree & tree,
        const std::vector<int32_t> & path,
        const std::vector<llama_seq_id> & seqs,
        llama_pos pos0) {
    GGML_ASSERT((int32_t) seqs.size() >= tree.n_branches());

    const int32_t n_path = (int32_t) path.size();

    // the leading nodes of the path that lie on branch 0 are already in seqs[0]
    int32_t n_keep = 0;
    while (n_keep < n_path && tree.branches[path[n_keep]][0] == 0) {
        n_keep++;
    }

    mem.seq_rm(seqs[0], pos0 + 1 + n_keep, -1);

    if (n_keep < n_path) {
        // the rest of the path is in every branch through its last node
        const int32_t b = tree.branches[path[n_path - 1]][0];

        mem.seq_cp(seqs[b], seqs[0], pos0 + 1 + n_keep, pos0 + 1 + n_path);
    }

    // the branches only differ from seqs[0] from the root on
    for (size_t b = 1; b < seqs.size(); ++b) {
        mem.seq_rm(seqs[b], pos0, -1);
        mem.seq_cp(seqs[0], seqs[b], pos0, pos0 + 1 + n_path);
    }
}
