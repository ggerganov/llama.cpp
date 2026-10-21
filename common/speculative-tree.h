#pragma once

#include "common.h"

#include <vector>

struct common_sampler;

// draft tree in breadth-first order (parent[i] < i), each root-to-leaf branch is decoded in its own sequence
struct common_speculative_tree {
    llama_tokens         tokens; // [n] drafted token per node
    std::vector<int32_t> parent; // [n] parent node, -1 for a child of the root

    // set by finalize()
    std::vector<int32_t>              depth;    // [n] 0 for a child of the root, node i is at pos0 + 1 + depth[i]
    std::vector<int32_t>              leaves;   // [n_branches] leaf of each branch, branch 0 follows the first children
    std::vector<std::vector<int32_t>> branches; // [n] branches through each node, ascending

    void clear();

    bool    empty() const { return tokens.empty(); }
    int32_t size()  const { return (int32_t) tokens.size(); }

    int32_t n_branches() const { return (int32_t) leaves.size(); }

    // returns the index of the new node, call finalize() after the last one
    int32_t add(llama_token id, int32_t i_parent);

    // a chain of draft tokens
    void from_chain(const llama_tokens & draft);

    // keep the first n nodes
    void truncate(int32_t n);

    void finalize();

    // the child of node (-1 = the root) with token id, -1 if there is none
    int32_t child(int32_t node, llama_token id) const;

    // the sequences of the branches through node (-1 = the root), seqs[b] is the sequence of branch b
    std::vector<llama_seq_id> node_seqs(int32_t node, const std::vector<llama_seq_id> & seqs) const;
};

// sequence of branch b of seq_id: branch 0 is seq_id itself, the others follow the n_seq sequences
llama_seq_id common_speculative_branch_seq(uint32_t n_seq, int32_t n_branches, llama_seq_id seq_id, int32_t b);

// make the sequences of the other branches copies of seqs[0]
void common_speculative_tree_fork(const common_memory & mem, const std::vector<llama_seq_id> & seqs);

// check that the other branches still span the same positions as seqs[0]
bool common_speculative_tree_synced(const common_memory & mem, const std::vector<llama_seq_id> & seqs);

// greedy walk from the root (idxs[0], node i at idxs[1 + i]) along the sampled tokens, path gets the accepted nodes
std::vector<llama_token> common_speculative_tree_sample_and_accept(
        common_sampler * smpl,
        llama_context * ctx,
        const common_speculative_tree & tree,
        const std::vector<int32_t> & idxs,
        std::vector<int32_t> & path);

// keep the accepted path in seqs[0] and make the other branches copies of it again, pos0 is the position of the root
void common_speculative_tree_commit(
        const common_memory & mem,
        const common_speculative_tree & tree,
        const std::vector<int32_t> & path,
        const std::vector<llama_seq_id> & seqs,
        llama_pos pos0);
