#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s \\\n"
            "       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--process-output] \\\n"
            "       [--no-ppl] [--chunk 123] [--output-frequency 10] [--save-frequency 0] \\\n"
            "       [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...]\n" , argv[0]);
    LOG("\n");
}

static bool str_remove_suffix(std::string & str, const std::string & suffix) {
    bool has_suffix = str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), str.size(), suffix) == 0;
    if (has_suffix) {
        str = str.substr(0, str.size() - suffix.size());
    }
    return has_suffix;
}

static const char * const LLM_KV_IMATRIX_DATASET     = "imatrix.dataset";
static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

struct Stats {
    std::vector<float>   values;
    std::vector<int64_t> counts;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(common_params params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void save_imatrix(int32_t n_chunk = -1) const;
    bool load_imatrix(const char * file_name);
private:
    std::unordered_map<std::string, Stats> m_stats;
    common_params                          m_params;
    std::mutex                             m_mutex;
    int32_t                                m_last_chunk = 0;
    std::vector<float>                     m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
};

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        // why are small batches ignored (<16 tokens)?
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(wname.substr(0, 4) == "blk." || (m_params.process_output && wname == "output.weight"))) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        m_src1_data.resize(ggml_nelements(src1));
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, ggml_nbytes(src1));
    }

    const float * data = is_host ? (const float *) src1->data : m_src1_data.data();

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor * ids = t->src[2];
        const int n_as = src0->ne[2];
        const int n_ids = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto & e = m_stats[wname];

        if (e.counts.size() == 1 && n_as > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(n_as, e.counts[0]);
        }
        if (e.values.empty()) {
            e.values.resize(src1->ne[0]*n_as, 0);
            e.counts.resize(n_as, 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]*n_as) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)src1->ne[0]*n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        else if (e.counts.size() != (size_t)n_as) {
            LOG_ERR("Oops: inconsistent expert count for %s (%d vs %d)\n", wname.c_str(), (int)e.counts.size(), (int)n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex*src1->ne[0];

            for (int idx = 0; idx < n_ids; ++idx) {
                for (int row = 0; row < (int)src1->ne[2]; ++row) {
                    const int excur = *(const int32_t *) (m_ids.data() + row*ids->nb[1] + idx*ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as); // sanity check

                    if (excur != ex) continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float * x = (const float *)((const char *)data + i11*src1->nb[1] + i12*src1->nb[2]);

                    e.counts[ex]++;

                    for (int j = 0; j < (int)src1->ne[0]; ++j) {
                        e.values[e_start + j] = std::fma(x[j], x[j], e.values[e_start + j]);
                        if (!std::isfinite((float)e.values[e_start + j])) {
                            LOG_ERR("%f detected in %s\n", (float)e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
            const int32_t n_chunk = e.counts[ex] / (m_params.n_ctx / m_params.n_parallel);
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    } else {
        auto & e = m_stats[wname];
        if (e.values.empty()) {
            e.values.resize(src1->ne[0], 0);
            e.counts.resize(1, 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)src1->ne[0]);
            exit(1); //GGML_ABORT("fatal error");
        }
        else if (e.counts.size() != 1) {
            LOG_ERR("Oops: inconsistent expert count for %s (%d vs %d)\n", wname.c_str(), (int)e.counts.size(), 1);
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->type);
        // TODO: higher dimensions
        for (int row = 0; row < (int)src1->ne[1]; ++row) {
            const float * x = data + row * src1->ne[0];
            e.counts[0]++;
            for (int j = 0; j < (int)src1->ne[0]; ++j) {
                e.values[j] = std::fma(x[j], x[j], e.values[j]);
                if (!std::isfinite((float)e.values[j])) {
                    LOG_ERR("%f detected in %s\n", (float)e.values[j], wname.c_str());
                    exit(1);
                }
            }
        }
        const int32_t n_chunk = e.counts[0] / (m_params.n_ctx / m_params.n_parallel);
        if (n_chunk > m_last_chunk) {
            const int32_t chunk_step = n_chunk - m_last_chunk;
            m_last_chunk = n_chunk;
            if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                save_imatrix();
            }
            if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                save_imatrix(m_last_chunk);
            }
        }
    }

    return true;
}

void IMatrixCollector::save_imatrix(int32_t n_chunk) const {
    auto fname = m_params.out_file;
    if (fname.empty()) {
        fname = "imatrix.gguf";
    }

    if (n_chunk > 0) {
        fname += ".at_";
        fname += std::to_string(n_chunk);
    }

    // avoid writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    std::vector<std::string> to_store;
    size_t data_size = 0;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            LOG_WRN("%s: entry '%40s' has no data - skipping\n", __func__, kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%) - skipping\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
            continue;
        }

        to_store.push_back(kv.first);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.values.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.counts.size(), GGML_MEM_ALIGN);
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WRN("%s: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    struct ggml_init_params params = {
        /* .mem_size   = */ data_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct gguf_context * ctx_gguf = gguf_init_empty();

    gguf_set_val_str(ctx_gguf, "general.type", "imatrix");
    // Write the input filename to later on specify it in quantize
    gguf_set_val_str(ctx_gguf, LLM_KV_IMATRIX_DATASET, m_params.prompt_file.c_str());
    // Write the number of chunks the matrix was computed with
    gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT, m_last_chunk);
    gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE, m_params.n_ctx / m_params.n_parallel);

    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t nval = (int32_t) stat.values.size();
        const int32_t nmat = (int32_t) stat.counts.size();
        if (nval > 0) {
            struct ggml_tensor * sums = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
            struct ggml_tensor * counts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nmat);
            ggml_format_name(sums, "%s.sums", name.c_str());
            ggml_format_name(counts, "%s.counts", name.c_str());

            for (int32_t j = 0; j < nval; ++j) {
                ((float *) sums->data)[j] = (float) stat.values[j];
            }
            for (int32_t j = 0; j < nmat; ++j) {
                ((float *) counts->data)[j] = (float) stat.counts[j];
            }

            gguf_add_tensor(ctx_gguf, sums);
            gguf_add_tensor(ctx_gguf, counts);
        }
    }

    gguf_write_to_file(ctx_gguf, fname.c_str(), false);

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());

    gguf_free(ctx_gguf);
    ggml_free(ctx);
}

bool IMatrixCollector::load_imatrix(const char * file_name) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false, // the data is needed
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(file_name, meta_gguf_params);
    if (!ctx_gguf) {
        return false;
    }
    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, file_name);
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return false;
    }

    const std::string sums_suffix{".sums"};
    const std::string counts_suffix{".counts"};

    // Could re-use m_stats instead, but this allows
    // checking for completeness of *each* loaded imatrix file
    // and also makes it easier to re-use a similar implementation in quantize.cpp
    // Using an ordered map to get a deterministic iteration order.
    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;

        if (name.empty()) { continue; }

        if (str_remove_suffix(name, sums_suffix)) {
            // sums
            sums_counts_for[name].first = cur;
        } else if (str_remove_suffix(name, counts_suffix)) {
            // counts
            sums_counts_for[name].second = cur;
        } else {
            LOG_ERR("%s: invalid imatrix tensor name: %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }
    }

    for (const auto & sc : sums_counts_for) {
        const        std::string & name   = sc.first;
        const struct ggml_tensor * sums   = sc.second.first;
        const struct ggml_tensor * counts = sc.second.second;

        if (!sums || !counts) {
            LOG_ERR("%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        auto & e = m_stats[name];

        int64_t nval = ggml_nelements(sums);
        if (e.values.empty()) {
            e.values.resize(nval, 0);
        } else if ((size_t) nval != e.values.size()) {
            LOG_ERR("%s: mismatched sums size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) nval, e.values.size());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        int64_t ncounts = ggml_nelements(counts);
        if (e.counts.empty()) {
            e.counts.resize(ncounts, 0);
        } else if (e.counts.size() == 1 && ncounts > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(ncounts, e.counts[0]);
        } else if ((size_t) ncounts != e.counts.size()) {
            LOG_ERR("%s: mismatched counts size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) ncounts, e.counts.size());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        // Recreate the state as expected by save_imatrix()
        for (int64_t j = 0; j < nval; j++) {
            e.values[j] += ((const float *) sums->data)[j];
        }
        for (int64_t j = 0; j < ncounts; j++) {
            e.counts[j] += std::lround(((const float *) counts->data)[j]);
        }
    }
    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return true;
}

static IMatrixCollector g_collector;

static bool ik_collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_imatrix(t, ask, user_data);
}


struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float> & logits) {
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
    double & nll, double & nll2, float * logit_history, float * prob_history) {
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

static bool compute_imatrix(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    auto tim1 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    auto tim2 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (params.i_chunk > 0) {
        if (size_t((params.i_chunk + 2)*n_ctx) >= tokens.size()) {
            LOG_ERR("%s: there will be not enough tokens left after removing %d chunks\n", __func__, params.i_chunk);
            return false;
        }
        LOG_INF("%s: removing initial %d chunks (%d tokens)\n", __func__, params.i_chunk, params.i_chunk*n_ctx);
        tokens.erase(tokens.begin(), tokens.begin() + params.i_chunk*n_ctx);
    }

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens for a context of %d tokens\n", __func__, 2*n_ctx, n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n", __func__, tokens.size());
        return false;
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    if (params.compute_ppl) {
        logit_history.resize(tokens.size());
        prob_history.resize(tokens.size());
    }

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx*n_seq), 0, 1);

    std::vector<float> logits;
    if (params.compute_ppl && num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    LOG_INF("%s: computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", __func__, n_chunk, n_ctx, n_batch, n_seq);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_kv_cache_clear(ctx);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // clear the batch
            common_batch_clear(batch);

            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after eval
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }
                for (int k = 0; k < batch_size; ++k) {
                    // NOTE: specifying all logits to get activations for the output.weight tensor
                    //       and also for the perplexity calculation.
                    // TODO: only get outputs when (params.process_output || params.compute_ppl)
                    //       (not possible when this skips FFN computation of the last layer)
                    common_batch_add(batch, tokens[seq_start + k], j*n_batch + k, { seq }, true);
                }
  
                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return false;
            }

            if (params.compute_ppl && num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
            }
        }


        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk / n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        if (params.compute_ppl) {
            const int first = n_ctx/2;
            for (int seq = 0; seq < n_seq_batch; seq++) {
                const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx);

                llama_token * tokens_data = tokens.data() + start + seq*n_ctx + first;

                process_logits(n_vocab, all_logits + first*n_vocab,
                        tokens_data, n_ctx - 1 - first,
                        workers, nll, nll2,
                        logit_history.data() + start + seq*n_ctx + first,
                        prob_history.data()  + start + seq*n_ctx + first);

                count += n_ctx - first - 1;

                LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
            }
            fflush(stdout);

            logits.clear();
        }
    }

    LOG("\n");

    if (params.compute_ppl) {
        nll2 /= count;
        nll /= count;
        const double ppl = exp(nll);
        nll2 -= nll * nll;
        if (nll2 > 0) {
            nll2 = sqrt(nll2/(count-1));
            LOG("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
        } else {
            LOG("Unexpected negative standard deviation of log(prob)\n");
        }
    }

    llama_batch_free(batch);

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx = 512;
    params.logits_all = true;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    common_init();

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: imatrix tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    {
        const int32_t n_seq = std::max(1, params.n_batch / n_ctx);
        const int32_t n_kv = n_seq * n_ctx;

        params.n_parallel = n_seq;
        params.n_ctx      = n_kv;

        params.n_batch = std::min(params.n_batch, n_kv);
    }

    g_collector.set_params(params);

    for (const auto & in_file : params.in_files) {
        LOG_INF("%s : loading imatrix from '%s'\n", __func__, in_file.c_str());
        if (!g_collector.load_imatrix(in_file.c_str())) {
            LOG_ERR("%s : failed to load %s\n", __func__, in_file.c_str());
            return 1;
        }
    }

    if (params.in_files.size() > 1) {
        LOG_INF("%s : saving combined imatrix to '%s'\n", __func__, params.out_file.c_str());
        g_collector.save_imatrix();
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ik_collect_imatrix;
    params.cb_eval_user_data = NULL;
    params.warmup = false;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    if (params.prompt.empty()) {
        if (params.in_files.empty()) {
            LOG_ERR("Error: No prompt provided and no precomputed matrices (--in-file) to combine.\n");
            return 1;
        }
        LOG_INF("No prompt provided; combining precomputed matrices only.\n");
    } else {
        if (!compute_imatrix(ctx, params, n_ctx)) {
            return 1;
        }
    }


    g_collector.save_imatrix();

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
