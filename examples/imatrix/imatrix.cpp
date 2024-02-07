#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct Stats {
    std::vector<float> values;
    int ncall = 0;
};

struct StatParams {
    std::string ofile = "imatrix.dat";
    int         n_output_frequency = 10;
    int         verbosity = 1;
    int         keep_every = 0;
    bool        collect_output_weight = false;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_parameters(StatParams&& params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void save_imatrix() const;
    bool load_imatrix(const char * file_name, bool add);
    static bool load_imatrix(const char * file_name, std::unordered_map<std::string, Stats>& imatrix);
private:
    std::unordered_map<std::string, Stats> m_stats;
    StatParams                             m_params;
    std::mutex                             m_mutex;
    int                                    m_last_call = 0;
    std::vector<float>                     m_src1_data;
    std::vector<int>                       m_ids; // the expert ids from ggml_mul_mat_id
                                                  //
    void save_imatrix(const char * file_name) const;
    void keep_imatrix(int ncall) const;
};

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(strncmp(src0->name, "blk.", 4) == 0 || (m_params.collect_output_weight && strcmp(src0->name, "output.weight") == 0))) return false;
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

    if (t->op == GGML_OP_MUL_MAT_ID) {
        const int idx  = ((int32_t *) t->op_params)[0];
        const int n_as = ((int32_t *) t->op_params)[1];

        // the top-k selected expert ids are stored in the src0 tensor
        // for simplicity, always copy src0 to host, because it is small
        // take into account that src0 is not contiguous!
        GGML_ASSERT(src0->ne[1] == src1->ne[1]);
        GGML_ASSERT(n_as*ggml_nrows(src0)*sizeof(int) == GGML_PAD(ggml_nbytes(src0), n_as*sizeof(int)));
        m_ids.resize(ggml_nbytes(src0)/sizeof(int));
        ggml_backend_tensor_get(src0, m_ids.data(), 0, ggml_nbytes(src0));

        // loop over all possible experts, regardless if they are used or not in the batch
        // this is necessary to guarantee equal number of "ncall" for each tensor
        for (int ex = 0; ex < n_as; ++ex) {
            src0 = t->src[2 + ex];
            auto& e = m_stats[src0->name];
            if (e.values.empty()) {
                e.values.resize(src1->ne[0], 0);
            }
            else if (e.values.size() != (size_t)src1->ne[0]) {
                fprintf(stderr, "Oops: inconsistent size for %s (%d vs %d)\n", src0->name, (int)e.values.size(), (int)src1->ne[0]);
                exit(1); //GGML_ASSERT(false);
            }
            // NOTE: since we select top-k experts, the number of calls for the expert tensors will be k times larger
            //       using the following line, we can correct for that if needed
            //if (idx == t->src[0]->ne[0] - 1) ++e.ncall;
            ++e.ncall;
            if (m_params.verbosity > 1) {
                printf("%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_call, src0->name, ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->type);
            }
            for (int row = 0; row < (int)src1->ne[1]; ++row) {
                const int excur = m_ids[row*n_as + idx];
                GGML_ASSERT(excur >= 0 && excur < n_as); // sanity check
                if (excur != ex) continue;
                const float * x = data + row * src1->ne[0];
                for (int j = 0; j < (int)src1->ne[0]; ++j) {
                    e.values[j] += x[j]*x[j];
                }
            }
            if (e.ncall > m_last_call) {
                m_last_call = e.ncall;
                if (m_last_call % m_params.n_output_frequency == 0) {
                    save_imatrix();
                }
                if (m_params.keep_every > 0 && m_last_call%m_params.keep_every == 0) {
                    keep_imatrix(m_last_call);
                }
            }
        }
    } else {
        auto& e = m_stats[src0->name];
        if (e.values.empty()) {
            e.values.resize(src1->ne[0], 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]) {
            fprintf(stderr, "Oops: inconsistent size for %s (%d vs %d)\n", src0->name, (int)e.values.size(), (int)src1->ne[0]);
            exit(1); //GGML_ASSERT(false);
        }
        ++e.ncall;
        if (m_params.verbosity > 1) {
            printf("%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_call, src0->name, ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->type);
        }
        for (int row = 0; row < (int)src1->ne[1]; ++row) {
            const float * x = data + row * src1->ne[0];
            for (int j = 0; j < (int)src1->ne[0]; ++j) {
                e.values[j] += x[j]*x[j];
            }
        }
        if (e.ncall > m_last_call) {
            m_last_call = e.ncall;
            if (m_last_call % m_params.n_output_frequency == 0) {
                save_imatrix();
            }
            if (m_params.keep_every > 0 && m_last_call%m_params.keep_every == 0) {
                keep_imatrix(m_last_call);
            }
        }
    }

    return true;
}

void IMatrixCollector::save_imatrix() const {
    save_imatrix(m_params.ofile.empty() ? "imatrix.dat" : m_params.ofile.c_str());
}

void IMatrixCollector::keep_imatrix(int ncall) const {
    auto file_name = m_params.ofile;
    if (file_name.empty()) file_name = "imatrix.dat";
    file_name += ".at_";
    file_name += std::to_string(ncall);
    save_imatrix(file_name.c_str());
}

void IMatrixCollector::save_imatrix(const char * fname) const {
    std::ofstream out(fname, std::ios::binary);
    int n_entries = m_stats.size();
    out.write((const char*)&n_entries, sizeof(n_entries));
    for (auto& p : m_stats) {
        int len = p.first.size();
        out.write((const char*)&len, sizeof(len));
        out.write(p.first.c_str(), len);
        out.write((const char*)&p.second.ncall, sizeof(p.second.ncall));
        int nval = p.second.values.size();
        out.write((const char*)&nval, sizeof(nval));
        if (nval > 0) out.write((const char*)p.second.values.data(), nval*sizeof(float));
    }
    if (m_params.verbosity > 0) {
        fprintf(stderr, "\n%s: stored collected data after %d chunks in %s\n",__func__,m_last_call,fname);
    }
}

bool IMatrixCollector::load_imatrix(const char * imatrix_file, std::unordered_map<std::string, Stats>& imatrix_data) {
    std::ifstream in(imatrix_file, std::ios::binary);
    if (!in) {
        printf("%s: failed to open %s\n",__func__,imatrix_file);
        return false;
    }
    int n_entries;
    in.read((char*)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        printf("%s: no data in file %s\n", __func__, imatrix_file);
        return false;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_as_vec(len+1);
        in.read((char *)name_as_vec.data(), len);
        if (in.fail()) {
            printf("%s: failed reading name for entry %d from %s\n",__func__,i+1,imatrix_file);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto& e = imatrix_data[std::move(name)];
        int ncall;
        in.read((char*)&ncall, sizeof(ncall));
        int nval;
        in.read((char *)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            printf("%s: failed reading number of values for entry %d\n",__func__,i);
            imatrix_data = {};
            return false;
        }
        e.values.resize(nval);
        in.read((char*)e.values.data(), nval*sizeof(float));
        if (in.fail()) {
            printf("%s: failed reading data for entry %d\n",__func__,i);
            imatrix_data = {};
            return false;
        }
        e.ncall = ncall;
    }
    return true;
}

bool IMatrixCollector::load_imatrix(const char * file_name, bool add) {
    if (!add) {
        m_stats.clear();
    }
    return load_imatrix(file_name, m_stats);
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

static bool compute_imatrix(llama_context * ctx, const gpt_params & params, bool compute_ppl, int from_chunk) {

    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    const int n_ctx = llama_n_ctx(ctx);

    auto tim1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);

    auto tim2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (from_chunk > 0) {
        if (size_t((from_chunk + 2)*n_ctx) >= tokens.size()) {
            fprintf(stderr, "%s: there will be not enough tokens left after removing %d chunks\n", __func__, from_chunk);
            return false;
        }
        fprintf(stderr, "%s: removing initial %d chunks (%d tokens)\n", __func__, from_chunk, from_chunk*n_ctx);
        tokens.erase(tokens.begin(), tokens.begin() + from_chunk*n_ctx);
    }

    if (int(tokens.size()) < 2*n_ctx) {
        fprintf(stderr, "%s: you need at least %d tokens for a context of %d tokens\n",__func__,2*n_ctx,
                n_ctx);
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return false;
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    if (compute_ppl) {
        logit_history.resize(tokens.size());
        prob_history.resize(tokens.size());
    }

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    fprintf(stderr, "%s: computing over %d chunks with batch_size %d\n", __func__, n_chunk, n_batch);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;

    std::vector<float> logits;
    if (compute_ppl && num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        std::vector<float> logits;

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
                return false;
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            if (compute_ppl && num_batches > 1) {
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

        if (compute_ppl) {
            const int first = n_ctx/2;
            const auto all_logits = num_batches > 1 ? logits.data() : llama_get_logits(ctx);
            process_logits(n_vocab, all_logits + first*n_vocab, tokens.data() + start + first, n_ctx - 1 - first,
                    workers, nll, nll2, logit_history.data() + start + first, prob_history.data() + start + first);
            count += n_ctx - first - 1;

            printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
            fflush(stdout);

            logits.clear();
        }
    }
    printf("\n");

    if (compute_ppl) {
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
    }

    return true;
}

int main(int argc, char ** argv) {

    StatParams sparams;
    std::string prev_result_file;
    std::string combine_files;
    bool compute_ppl = true;
    int  from_chunk  = 0;
    std::vector<char*> args;
    args.push_back(argv[0]);
    int iarg = 1;
    for (; iarg < argc-1; ++iarg) {
        std::string arg{argv[iarg]};
        if (arg == "-o" || arg == "--output-file") {
            sparams.ofile = argv[++iarg];
        }
        else if (arg == "-ofreq" || arg == "--output-frequency") {
            sparams.n_output_frequency = std::stoi(argv[++iarg]);
        }
        else if (arg == "-ow" || arg == "--output-weight") {
            sparams.collect_output_weight = std::stoi(argv[++iarg]);
        }
        else if (arg == "--verbosity") {
            sparams.verbosity = std::stoi(argv[++iarg]);
        } else if (arg == "--no-ppl") {
            compute_ppl = false;
        } else if (arg == "--keep-imatrix") {
            sparams.keep_every = std::stoi(argv[++iarg]);
        } else if (arg == "--continue-from") {
            prev_result_file = argv[++iarg];
        } else if (arg == "--combine") {
            combine_files = argv[++iarg];
        }
        else if (arg == "--from-chunk") {
            from_chunk = std::stoi(argv[++iarg]);
        } else {
            args.push_back(argv[iarg]);
        }
    }
    if (iarg < argc) {
        std::string arg{argv[iarg]};
        if (arg == "--no-ppl") {
            compute_ppl = false;
        } else {
            args.push_back(argv[iarg]);
        }
    }

    g_collector.set_parameters(std::move(sparams));

    if (!combine_files.empty()) {
        std::vector<std::string> files;
        size_t pos = 0;
        while (true) {
            auto new_pos = combine_files.find(',', pos);
            if (new_pos != std::string::npos) {
                files.emplace_back(combine_files.substr(pos, new_pos - pos));
                pos = new_pos + 1;
            } else {
                files.emplace_back(combine_files.substr(pos));
                break;
            }
        }
        if (files.size() < 2) {
            fprintf(stderr, "You must provide at least two comma separated files to use --combine\n");
            return 1;
        }
        printf("Combining the following %d files\n", int(files.size()));
        for (auto& file : files) {
            printf("    %s\n", file.c_str());
            if (!g_collector.load_imatrix(file.c_str(), true)) {
                fprintf(stderr, "Failed to load %s\n", file.c_str());
                return 1;
            }
        }
        g_collector.save_imatrix();
        return 0;
    }

    if (!prev_result_file.empty()) {
        if (!g_collector.load_imatrix(prev_result_file.c_str(), false)) {
            fprintf(stderr, "=============== Failed to load %s\n", prev_result_file.c_str());
            return 1;
        }
    }

    gpt_params params;
    params.n_batch = 512;
    if (!gpt_params_parse(args.size(), args.data(), params)) {
        return 1;
    }

    params.logits_all = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

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

    llama_model_params mparams = llama_model_params_from_gpt_params(params);

    llama_model * model = llama_load_model_from_file(params.model.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    llama_context_params cparams = llama_context_params_from_gpt_params(params);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    cparams.cb_eval = ik_collect_imatrix;
    cparams.cb_eval_user_data = NULL;

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: unable to create context\n", __func__);
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

    bool OK = compute_imatrix(ctx, params, compute_ppl, from_chunk);
    if (!OK) {
        return 1;
    }

    g_collector.save_imatrix();

    llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
