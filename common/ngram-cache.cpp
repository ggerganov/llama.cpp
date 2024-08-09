#include "ngram-cache.h"
#include "common.h"
#include "log.h"

#include <cstdint>
#include <fstream>

void llama_ngram_cache_update(llama_ngram_cache & ngram_cache, int ngram_min, int ngram_max,
                              std::vector<llama_token> & inp, int nnew, bool print_progress) {
    const int64_t t_start_ms = ggml_time_ms();
    const int64_t inp_size = inp.size();

    const int64_t n_todo = inp_size * (ngram_max - ngram_min + 1);
    int64_t n_done = 0;

    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            llama_ngram ngram(&inp[ngram_start], ngram_size);
            const llama_token token = inp[i];

            llama_ngram_cache::iterator part_it = ngram_cache.find(ngram);
            if (part_it == ngram_cache.end()) {
                llama_ngram_cache_part part;
                part.emplace(token, 1);
                ngram_cache.emplace(ngram, part);
            } else {
                llama_ngram_cache_part::iterator token_count_it = part_it->second.find(token);
                if (token_count_it == part_it->second.end()) {
                    part_it->second.emplace(token, 1);
                } else {
                    token_count_it->second++;
                }
            }
            ++n_done;

            if (print_progress && n_done % 10000000 == 0) {
                const int64_t t_now_ms = ggml_time_ms();
                const int64_t eta_ms   = (inp_size*(ngram_max-ngram_min+1) - n_done) * (t_now_ms - t_start_ms) / n_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

                fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, n_done, n_todo, eta_min, eta_s);
            }
        }
    }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static llama_token get_token(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, const size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

// Sample size and percentage must meet these thresholds to be added to the draft tree:
constexpr int    draft_min_sample_size_lax[LLAMA_NGRAM_MAX] = { 1,  1,  1,  1};
constexpr int        draft_min_percent_lax[LLAMA_NGRAM_MAX] = {20, 20, 10, 10};
constexpr int draft_min_sample_size_strict[LLAMA_NGRAM_MAX] = { 4,  3,  2,  2};
constexpr int     draft_min_percent_strict[LLAMA_NGRAM_MAX] = {50, 50, 50, 50};

struct draft_candidate {
    llama_draft_t draft;
    float nll;
    int nsampled;
};

struct compare_draft_candidate {
    bool operator()(const draft_candidate & a, const draft_candidate & b){
        if (a.nsampled > b.nsampled) {
            return true;
        }
        if (a.nsampled < b.nsampled) {
            return false;
        }
        return a.nll < b.nll;
    }
};

// Helper function that tries to draft tokens from only the static ngram cache:
static void try_draft(
    llama_ngram_cache & nc_static, const llama_ngram & ngram_static,
    const int * min_sample_size, const int * min_percent, const draft_candidate & cp,
    const int ngram_min, std::vector<draft_candidate> & drafts_new) {

    const int nsc = (ngram_min + LLAMA_NGRAM_STATIC) - (cp.draft.size() - 1);
    if (nsc < (ngram_min + LLAMA_NGRAM_STATIC + 1)/2) {
        return;
    }

    llama_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
    if (part_static_it == nc_static.end()) {
        return;
    }
    const llama_ngram_cache_part part_static = part_static_it->second;

    int sum_count_static  = 0;

    for (std::pair<llama_token, int> token_count_static : part_static) {
        const int32_t count_static  = token_count_static.second;

        sum_count_static += count_static;
    }

    for (std::pair<llama_token, int> token_count_static : part_static) {
        const llama_token token = token_count_static.first;
        const int32_t count_static  = token_count_static.second;

        if (sum_count_static < min_sample_size[LLAMA_NGRAM_STATIC-1]) {
            continue;
        }
        if (100*count_static < min_percent[LLAMA_NGRAM_STATIC-1]*sum_count_static) {
            continue;;
        }

        draft_candidate cc;
        for (const llama_token & t : cp.draft) {
            cc.draft.push_back(t);
        }
        cc.draft.push_back(token);
        cc.nll = cp.nll - logf(1.0f*count_static/sum_count_static);
        cc.nsampled = nsc;

        bool duplicate = false;
        for (const draft_candidate & co : drafts_new) {
            if (co.draft == cc.draft) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }

        drafts_new.push_back(cc);
    }
}

// Try to draft tokens from primary cache (context/dynamic), validate with static cache:
static void try_draft(
    llama_ngram_cache & nc_primary, const std::vector<llama_ngram> & ngrams_primary, llama_ngram_cache_part & part_static,
    const int * min_sample_size, const int * min_percent, const draft_candidate & cp,
    const int ngram_min, std::vector<draft_candidate> & drafts_new) {

    for (int i = ngrams_primary.size()-1; i >= 0; --i) {
        const int nsc = (ngram_min + i) - (cp.draft.size() - 1);
        if (nsc < (ngram_min + i + 1)/2) {
            break;
        }

        const llama_ngram ngram_primary = ngrams_primary[i];

        llama_ngram_cache::iterator part_primary_it = nc_primary.find(ngram_primary);
        if (part_primary_it == nc_primary.end()) {
            continue;
        }
        const llama_ngram_cache_part part_primary = part_primary_it->second;

        int sum_count_primary = 0;
        int sum_count_prod    = 0;

        for (std::pair<llama_token, int> token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;

            llama_ngram_cache_part::iterator token_count_static_it = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static  = token_count_static_it != part_static.end() ? 100*token_count_static_it->second : 1;

            sum_count_primary += count_primary;
            sum_count_prod    += count_primary*count_static;
        }

        for (std::pair<llama_token, int> token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;

            llama_ngram_cache_part::iterator token_count_static_it = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static  = token_count_static_it != part_static.end() ? 100*token_count_static_it->second : 1;
            const int32_t count_prod    = count_primary*count_static;

            if (sum_count_primary < min_sample_size[i]) {
                continue;
            }

            if (100*count_prod < min_percent[i]*sum_count_prod) {
                continue;
            }

            draft_candidate cc;
            for (const llama_token & t : cp.draft) {
                cc.draft.push_back(t);
            }
            cc.draft.push_back(token);
            cc.nll = cp.nll - logf(1.0f*count_prod/sum_count_prod);
            cc.nsampled = nsc;

            bool duplicate = false;
            for (const draft_candidate & co : drafts_new) {
                if (co.draft == cc.draft) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }

            drafts_new.push_back(cc);
        }
    }
}

void llama_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<std::vector<llama_token>> & drafts, int n_draft, int ngram_min, int ngram_max,
    llama_ngram_cache & nc_context, llama_ngram_cache & nc_dynamic, llama_ngram_cache & nc_static
) {
    if (n_draft == 0) {
        return;
    }

    GGML_ASSERT(drafts.size() == 1);
    GGML_ASSERT(drafts[0].size() == 1);
    const int inp_size = inp.size();

    if (inp_size < std::max(ngram_max, LLAMA_NGRAM_STATIC)) {
        return;
    }

    // While building the tree, store drafts with potential children in a heap:
    std::vector<draft_candidate> drafts_wip;

    {
        draft_candidate candidate;
        candidate.draft.push_back(drafts[0][0]);
        candidate.nll = 0.0f;
        candidate.nsampled = LLAMA_NGRAM_MAX;
        drafts_wip.push_back(candidate);
    }

    drafts.clear();
    int i_draft = 0;

    // Temporarily hold new drafts in vector, only add part of them in the last iteration to exactly meet n_draft.
    std::vector<draft_candidate> drafts_new;

    while (i_draft + ((int) drafts_new.size()) < n_draft && !(drafts_wip.empty() && drafts_new.empty())) {
        for (const draft_candidate & ndc : drafts_new) {
            drafts_wip.push_back(ndc);
            std::push_heap(drafts_wip.begin(), drafts_wip.end(), compare_draft_candidate());
            i_draft++;
        }
        drafts_new.clear();

        std::pop_heap(drafts_wip.begin(), drafts_wip.end(), compare_draft_candidate());
        const draft_candidate cp = drafts_wip.back(); // cp = candidate parent
        drafts_wip.pop_back();

        const int ngram_start_static = inp_size-LLAMA_NGRAM_STATIC + cp.draft.size()-1;
        llama_ngram ngram_static;
        for (int j = ngram_start_static; j < ngram_start_static + LLAMA_NGRAM_STATIC; ++j) {
            ngram_static.tokens[j-ngram_start_static] = get_token(inp, cp.draft, j);
        }
        llama_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
        llama_ngram_cache_part part_static;
        if (part_static_it != nc_static.end()) {
            part_static = part_static_it->second;
        }

        // cd = context + dynamic
        std::vector<llama_ngram> ngrams_cd;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max; ++ngram_size_cd) {
            const int ngram_start_cd = inp_size-ngram_size_cd + cp.draft.size()-1;
            llama_ngram ngram_cd;
            for (int j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                ngram_cd.tokens[j-ngram_start_cd] = get_token(inp, cp.draft, j);
            }
            ngrams_cd.push_back(ngram_cd);
        }

        try_draft(nc_context, ngrams_cd,    part_static, draft_min_sample_size_lax,    draft_min_percent_lax,    cp, ngram_min, drafts_new);
        try_draft(nc_dynamic, ngrams_cd,    part_static, draft_min_sample_size_strict, draft_min_percent_lax,    cp, ngram_min, drafts_new);
        try_draft(nc_static,  ngram_static,              draft_min_sample_size_strict, draft_min_percent_strict, cp, ngram_min, drafts_new);

        if (drafts_new.empty()) {
            drafts.push_back(cp.draft);
            i_draft++;
        }
    }

    for (const draft_candidate & dc : drafts_wip) { // dc = draft child
        drafts.push_back(dc.draft);
    }

    std::sort(drafts_new.begin(), drafts_new.end(), compare_draft_candidate());

    for (const draft_candidate & dc : drafts_new) {
        drafts.push_back(dc.draft);
        i_draft++;

        if (i_draft >= n_draft) {
            break;
        }
    }
}

void llama_ngram_cache_save(llama_ngram_cache & ngram_cache, std::string & filename) {
    std::ofstream file_out(filename, std::ios::binary);
    for (std::pair<llama_ngram, llama_ngram_cache_part> item : ngram_cache) {
        const llama_ngram      ngram        = item.first;
        llama_ngram_cache_part token_counts = item.second;
        GGML_ASSERT(!token_counts.empty());
        const int32_t ntokens = token_counts.size();
        GGML_ASSERT(ntokens > 0);

        file_out.write(reinterpret_cast<const char *>(&ngram),   sizeof(llama_ngram));
        file_out.write(reinterpret_cast<const char *>(&ntokens), sizeof(int32_t));
        for (std::pair<llama_token, int32_t> item2 : token_counts) {
            const llama_token token = item2.first;
            const int32_t     count = item2.second;
            GGML_ASSERT(count > 0);

            file_out.write(reinterpret_cast<const char *>(&token), sizeof(llama_token));
            file_out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    }

}

llama_ngram_cache llama_ngram_cache_load(std::string & filename) {
    std::ifstream hashmap_file(filename, std::ios::binary);
    if (!hashmap_file) {
        throw std::ifstream::failure("Unable to open file " + filename);
    }
    llama_ngram_cache ngram_cache;

    llama_ngram ngram;
    int32_t     ntokens;
    llama_token token;
    int32_t     count;

    char * ngramc   = reinterpret_cast<char*>(&ngram);
    char * ntokensc = reinterpret_cast<char*>(&ntokens);
    char * tokenc   = reinterpret_cast<char*>(&token);
    char * countc   = reinterpret_cast<char*>(&count);
    while(hashmap_file.read(ngramc, sizeof(llama_ngram))) {
        GGML_ASSERT(!hashmap_file.eof());
        GGML_ASSERT(hashmap_file.read(ntokensc, sizeof(int32_t)));
        GGML_ASSERT(ntokens > 0);
        llama_ngram_cache_part token_counts;

        for (int i = 0; i < ntokens; ++i) {
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(tokenc, sizeof(llama_token)));
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(countc, sizeof(int32_t)));
            GGML_ASSERT(count > 0);
            token_counts.emplace(token, count);
        }

        ngram_cache.emplace(ngram, token_counts);
    }
    GGML_ASSERT(hashmap_file.eof());

    return ngram_cache;
}

void llama_ngram_cache_merge(llama_ngram_cache & ngram_cache_target, llama_ngram_cache & ngram_cache_add) {
    for (std::pair<llama_ngram, llama_ngram_cache_part> ngram_part : ngram_cache_add) {
        const llama_ngram      ngram = ngram_part.first;
        llama_ngram_cache_part  part = ngram_part.second;

        llama_ngram_cache::iterator part_merged_it = ngram_cache_target.find(ngram);
        if (part_merged_it == ngram_cache_target.end()) {
            ngram_cache_target.emplace(ngram, part);
            continue;
        }

        for (std::pair<llama_token, int32_t> token_count : part) {
            const llama_token token = token_count.first;
            const int32_t     count = token_count.second;
            GGML_ASSERT(count > 0);

            llama_ngram_cache_part::iterator token_count_merged_it = part_merged_it->second.find(token);
            if (token_count_merged_it == part_merged_it->second.end()) {
                part_merged_it->second.emplace(token, count);
                continue;
            }

            token_count_merged_it->second += count;
        }
    }
}
