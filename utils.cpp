#include "utils.h"
#define QK 32

#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-n" || arg == "--n_predict") {
            params.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--top_k") {
            params.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top_p") {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--temp") {
            params.temp = std::stof(argv[++i]);
        } else if (arg == "-b" || arg == "--batch_size") {
            params.n_batch = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void gpt_print_usage(int argc, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "\n");
}

std::string gpt_random_prompt(std::mt19937 & rng) {
    const int r = rng() % 10;
    switch (r) {
        case 0: return "So";
        case 1: return "Once upon a time";
        case 2: return "When";
        case 3: return "The";
        case 4: return "After";
        case 5: return "If";
        case 6: return "import";
        case 7: return "He";
        case 8: return "She";
        case 9: return "They";
        default: return "To";
    }

    return "The";
}

void replace(std::string & str, const std::string & needle, const std::string & replacement) {
    size_t pos = 0;
    while ((pos = str.find(needle, pos)) != std::string::npos) {
        str.replace(pos, needle.length(), replacement);
        pos += replacement.length();
    }
}

std::map<std::string, int32_t> json_parse(const std::string & fname) {
    std::map<std::string, int32_t> result;

    // read file into string
    std::string json;
    {
        std::ifstream ifs(fname);
        if (!ifs) {
            fprintf(stderr, "Failed to open %s\n", fname.c_str());
            exit(1);
        }

        json = std::string((std::istreambuf_iterator<char>(ifs)),
                (std::istreambuf_iterator<char>()));
    }

    if (json[0] != '{') {
        return result;
    }

    // parse json
    {
        bool has_key  = false;
        bool in_token = false;

        std::string str_key = "";
        std::string str_val = "";

        int n = json.size();
        for (int i = 1; i < n; ++i) {
            if (!in_token) {
                if (json[i] == ' ') continue;
                if (json[i] == '"') {
                    in_token = true;
                    continue;
                }
            } else {
                if (json[i] == '\\' && i+1 < n) {
                    if (has_key == false) {
                        str_key += json[i];
                    } else {
                        str_val += json[i];
                    }
                    ++i;
                } else if (json[i] == '"') {
                    if (has_key == false) {
                        has_key = true;
                        ++i;
                        while (json[i] == ' ') ++i;
                        ++i; // :
                        while (json[i] == ' ') ++i;
                        if (json[i] != '\"') {
                            while (json[i] != ',' && json[i] != '}') {
                                str_val += json[i++];
                            }
                            has_key = false;
                        } else {
                            in_token = true;
                            continue;
                        }
                    } else {
                        has_key = false;
                    }

                    ::replace(str_key, "\\u0120", " " ); // \u0120 -> space
                    ::replace(str_key, "\\u010a", "\n"); // \u010a -> new line
                    ::replace(str_key, "\\\"",    "\""); // \\\"   -> "

                    try {
                        result[str_key] = std::stoi(str_val);
                    } catch (...) {
                        //fprintf(stderr, "%s: ignoring key '%s' with value '%s'\n", fname.c_str(), str_key.c_str(), str_val.c_str());

                    }
                    str_key = "";
                    str_val = "";
                    in_token = false;
                    continue;
                }
                if (has_key == false) {
                    str_key += json[i];
                } else {
                    str_val += json[i];
                }
            }
        }
    }

    return result;
}

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    // find the longest tokens that form the words:
    std::vector<gpt_vocab::id> tokens;
    for (const auto & word : words) {
        if (word.size() == 0) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            while (j > i) {
                auto it = vocab.token_to_id.find(word.substr(i, j-i));
                if (it != vocab.token_to_id.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    break;
                }
                --j;
            }
            if (i == n) {
                break;
            }
            if (j == i) {
                auto sub = word.substr(i, 1);
                if (vocab.token_to_id.find(sub) != vocab.token_to_id.end()) {
                    tokens.push_back(vocab.token_to_id.at(sub));
                } else {
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                }
                ++i;
            }
        }
    }

    return tokens;
}

std::vector<gpt_vocab::id> llama_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos) {
    //auto res = gpt_tokenize(vocab, text);

    //if (bos) {
    //    res.insert(res.begin(), 1); // TODO: replace with vocab.bos
    //}

    std::vector<gpt_vocab::id> res;

    if (bos) {
        res.push_back(1); // TODO: replace with vocab.bos
    }

     //find the longest token that matches the text
    int pos = 0;
    while (true) {
        int l = 0;
        int t = 0;
        for (const auto & kv : vocab.id_to_token) {
            if (kv.second.size() < l) continue;
            if (kv.second.size() > text.size() - pos) continue;
            if (text.substr(pos, kv.second.size()) == kv.second) {
                l = kv.second.size();
                t = kv.first;
            }
        }

        if (l == 0) {
            break;
        }

        res.push_back(t);
        pos += l;
    }

    return res;
}

bool gpt_vocab_init(const std::string & fname, gpt_vocab & vocab) {
    printf("%s: loading vocab from '%s'\n", __func__, fname.c_str());

    vocab.token_to_id = ::json_parse(fname);

    for (const auto & kv : vocab.token_to_id) {
        vocab.id_to_token[kv.second] = kv.first;
    }

    printf("%s: vocab size = %d\n", __func__, (int) vocab.token_to_id.size());

    // print the vocabulary
    //for (auto kv : vocab.token_to_id) {
    //    printf("'%s' -> %d\n", kv.first.data(), kv.second);
    //}

    return true;
}

gpt_vocab::id gpt_sample_top_k_top_p(
        const gpt_vocab & vocab,
        const float * logits,
        int    top_k,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            logits_id.push_back(std::make_pair(logits[i]*scale, i));
        }
    }

    // find the top K tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                top_k = i + 1;
                probs.resize(top_k);
                logits_id.resize(top_k);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) probs.size(); i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}

gpt_vocab::id llama_sample_top_p(
        const gpt_vocab & vocab,
        const float * logits,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            logits_id.push_back(std::make_pair(logits[i]*scale, i));
        }
    }

    std::sort(
            logits_id.begin(),
            logits_id.end(),
            [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
        return a.first > b.first;
    });

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}


size_t ggml_quantize_q4_0(float * src, void * dst, int n, int k, int qk, int64_t * hist) {
    const int nb = k / qk;
    const size_t bs = (sizeof(float) + sizeof(uint8_t)*qk/2);
    const size_t row_size = nb*bs;

    assert(k % qk == 0);

    uint8_t pp[QK/2];

    char * pdst = (char *) dst;

    for (int j = 0; j < n; j += k) {
        uint8_t * pd = (uint8_t *) (pdst + (j/k)*row_size + 0*bs);
        uint8_t * pb = (uint8_t *) (pdst + (j/k)*row_size + 0*bs + sizeof(float));

        for (int i = 0; i < nb; i++) {
            float amax = 0.0f; // absolute max

            {
                for (int l = 0; l < qk; l++) {
                    const float v = src[j + i*qk + l];
                    amax = std::max(amax, fabsf(v));
                }

                const float d = amax / ((1 << 3) - 1);
                const float id = d ? 1.0f/d : 0.0f;

                *(float *) pd = d;
                pd += bs;

                for (int l = 0; l < qk; l += 2) {
                    const float v0 = (src[j + i*qk + l + 0])*id;
                    const float v1 = (src[j + i*qk + l + 1])*id;

                    const uint8_t vi0 = ((int8_t) (round(v0))) + 8;
                    const uint8_t vi1 = ((int8_t) (round(v1))) + 8;

                    assert(vi0 >= 0 && vi0 < 16);
                    assert(vi1 >= 0 && vi1 < 16);

                    hist[vi0]++;
                    hist[vi1]++;

                    pp[l/2] = vi0 | (vi1 << 4);
                }

                memcpy(pb, pp, sizeof(pp));
                pb += bs;
            }
        }
    }

    return (n/k)*row_size;
}

size_t ggml_quantize_q4_1(float * src, void * dst, int n, int k, int qk, int64_t * hist) {
    const int nb = k / qk;
    const size_t row_size = nb*(2*sizeof(float) + sizeof(uint8_t)*qk/2);

    assert(k % qk == 0);

    uint8_t pp[QK/2];

    char * pdst = (char *) dst;

    for (int j = 0; j < n; j += k) {
        float   * pm = (float *)   (pdst + (j/k)*row_size);
        float   * pd = (float *)   (pm + nb);
        uint8_t * pb = (uint8_t *) (pd + nb);

        //printf("n = %d, k = %d, nb = %d, row_size = %d, j = %d, pm = %p, pd = %p, pb = %p\n", n, k, nb, row_size, j, pm, pd, pb);

        for (int i = 0; i < nb; i++) {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();

            {
                for (int l = 0; l < qk; l++) {
                    const float v = src[j + i*qk + l];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                const float d = (max - min) / ((1 << 4) - 1);
                const float id = d ? 1.0f/d : 0.0f;

                pm[i] = min;
                pd[i] = d;

                for (int l = 0; l < qk; l += 2) {
                    const float v0 = (src[j + i*qk + l + 0] - min)*id;
                    const float v1 = (src[j + i*qk + l + 1] - min)*id;

                    const uint8_t vi0 = round(v0);
                    const uint8_t vi1 = round(v1);

                    assert(vi0 >= 0 && vi0 < 16);
                    assert(vi1 >= 0 && vi1 < 16);

                    hist[vi0]++;
                    hist[vi1]++;

                    pp[l/2] = vi0 | (vi1 << 4);
                }

                memcpy(pb + i*qk/2, pp, sizeof(pp));
            }
        }
    }

    return (n/k)*row_size;
}
