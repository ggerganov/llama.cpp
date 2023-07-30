#include "gptneox-common.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <locale>
#include <codecvt>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// Function to check if the next argument exists
std::string get_next_arg(int& i, int argc, char** argv, const std::string& flag, gpt_params& params) {
    if (i + 1 < argc && argv[i + 1][0] != '-') {
        return argv[++i];
    } else {
        fprintf(stderr, "error: %s requires one argument.\n", flag.c_str());
        gpt_print_usage(argc, argv, params);
        exit(0);
    }
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers") {
            params.n_gpu_layers = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = get_next_arg(i, argc, argv, arg, params);
        } else if (arg == "-n" || arg == "--n_predict") {
            params.n_predict = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--top_k") {
            params.top_k = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--top_p") {
            params.top_p = std::stof(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--temp") {
            params.temp = std::stof(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--repeat-last-n") {
            params.repeat_last_n = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--repeat-penalty") {
            params.repeat_penalty = std::stof(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-b" || arg == "--batch_size") {
            params.n_batch= std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-m" || arg == "--model") {
            params.model = get_next_arg(i, argc, argv, arg, params);
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "-ip" || arg == "--interactive-port") {
            params.interactive = true;
            params.interactive_port = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, params);
            exit(0);
        } else if (arg == "-f" || arg == "--file") {
            get_next_arg(i, argc, argv, arg, params);
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-tt" || arg == "--token_test") {
            params.token_test = get_next_arg(i, argc, argv, arg, params);
        }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void gpt_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -ngl N, --gpu-layers N  number of layers to offload to GPU on supported models (default: %d)\n", params.n_gpu_layers);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        load prompt from a file\n");
    fprintf(stderr, "  -tt TOKEN_TEST, --token_test TOKEN_TEST\n");
    fprintf(stderr, "                        test tokenization\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)\n", (double)params.repeat_penalty);
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

std::string trim(const std::string & s) {
    std::regex e("^\\s+|\\s+$");
    return std::regex_replace(s, e, "");
}

std::string replace(const std::string & s, const std::string & from, const std::string & to) {
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

void gpt_vocab::add_special_token(const std::string & token) {
    special_tokens.push_back(token);
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

                    str_key = ::replace(str_key, "\\u0120", " " ); // \u0120 -> space
                    str_key = ::replace(str_key, "\\u010a", "\n"); // \u010a -> new line
                    str_key = ::replace(str_key, "\\\"",    "\""); // \\\"   -> "

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

std::string convert_to_utf8(const std::wstring & input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(input);
}


std::wstring convert_to_wstring(const std::string & input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(input);
}

void gpt_split_words(std::string str, std::vector<std::string>& words) {
    const std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::regex re(pattern);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
        for (auto x : m) {
            words.push_back(x);
        }
        str = m.suffix();
    }
}

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;

        // Generate the subpattern from the special_tokens vector if it's not empty
        if (!vocab.special_tokens.empty()) {
            const std::regex escape(R"([\[\\\^\$\.\|\?\*\+\(\)\{\}])");
            std::string special_tokens_subpattern;
            for (const auto & token : vocab.special_tokens) {
                if (!special_tokens_subpattern.empty()) {
                    special_tokens_subpattern += "|";
                }
                special_tokens_subpattern += std::regex_replace(token, escape, R"(\$&)");
            }

            std::regex re(special_tokens_subpattern);
            std::smatch m;
            // Split the text by special tokens.
            while (std::regex_search(str, m, re)) {
                // Split the substrings in-between special tokens into words.
                gpt_split_words(m.prefix(), words);
                // Add matched special tokens as words.
                for (auto x : m) {
                    words.push_back(x);
                }
                str = m.suffix();
            }
            // Remaining text without special tokens will be handled below.
        }

        gpt_split_words(str, words);
    }

    // find the longest token that forms each word in words:
    std::vector<gpt_vocab::id> tokens;
    for (const auto & word : words) {
        for (int i = 0; i < (int) word.size(); ){
            for (int j = word.size() - 1; j >= i; j--){
                auto cand = word.substr(i, j-i+1);
                auto it = vocab.token_to_id.find(cand);
                if (it != vocab.token_to_id.end()){ // word.substr(i, j-i+1) in vocab
                    tokens.push_back(it->second);
                    i = j + 1;
                    break;
                }
                else if (j == i){ // word.substr(i, 1) has no matching
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                    i++;
                }
            }
        }
    }

    return tokens;
}

std::vector<gpt_vocab::id> parse_tokens_from_string(const std::string& input, char delimiter) {
    std::vector<gpt_vocab::id> output;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        output.push_back(std::stoi(token));
    }

    return output;
}

std::map<std::string, std::vector<gpt_vocab::id>> extract_tests_from_file(const std::string & fpath_test){
    if (fpath_test.empty()){
        fprintf(stderr, "%s : No test file found.\n", __func__);
        return std::map<std::string, std::vector<gpt_vocab::id>>();
    }

    std::map<std::string, std::vector<gpt_vocab::id>> tests;

    auto fin = std::ifstream(fpath_test, std::ios_base::in);
    const char * delimeter = " => ";
    const char del_tok = ',';
    std::string line;
    while (std::getline(fin, line)) {
        size_t delimiterPos = line.find(delimeter);
        if (delimiterPos != std::string::npos) {
            std::string text = line.substr(0, delimiterPos);
            std::string s_tokens = line.substr(delimiterPos + std::strlen(delimeter));
            tests[text] = parse_tokens_from_string(s_tokens, del_tok);
        }
    }
    return tests;
}

void test_gpt_tokenizer(gpt_vocab & vocab, const std::string & fpath_test){
    std::map<std::string, std::vector<gpt_vocab::id>> tests = extract_tests_from_file(fpath_test);

    size_t n_fails = 0;

    for (const auto & test : tests) {
        std::vector<gpt_vocab::id> tokens = gpt_tokenize(vocab, test.first);

        if (tokens != test.second){
            n_fails++;

            // print out failure cases
            fprintf(stderr, "%s : failed test: '%s'\n", __func__, test.first.c_str());
            fprintf(stderr, "%s : tokens in hf:   ", __func__);
            for (const auto & t : test.second) {
                fprintf(stderr, "%s(%d), ", vocab.id_to_token[t].c_str(), t);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "%s : tokens in ggml: ", __func__);
            for (const auto & t : tokens) {
                fprintf(stderr, "%s(%d), ", vocab.id_to_token[t].c_str(), t);
            }
            fprintf(stderr, "\n");
        }
    }

    fprintf(stderr, "%s : %zu tests failed out of %zu tests.\n", __func__, n_fails, tests.size());
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

gpt_vocab::id gpt_sample_top_k_top_p_repeat(
        const gpt_vocab & vocab,
        const float * logits,
        const int32_t * last_n_tokens_data,
        size_t last_n_tokens_data_size,
        int    top_k,
        double top_p,
        double temp,
        int repeat_last_n,
        float repeat_penalty,
        std::mt19937 & rng) {

    int n_logits = vocab.id_to_token.size();

    const auto * plogits = logits;

    const auto last_n_tokens = std::vector<int32_t>(last_n_tokens_data, last_n_tokens_data + last_n_tokens_data_size);

    if (temp <= 0) {
        // select the token with the highest logit directly
        float max_logit = plogits[0];
        gpt_vocab::id max_id = 0;

        for (int i = 1; i < n_logits; ++i) {
            if (plogits[i] > max_logit) {
                max_logit = plogits[i];
                max_id = i;
            }
        }
        return max_id;
    }


    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const float scale = 1.0f/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (repeat_last_n > 0 && std::find(last_n_tokens.end()-repeat_last_n, last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (plogits[i] < 0.0f) {
                    logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(plogits[i]*scale, i));
            }
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

//    printf("\n");
//    for (int i = 0; i < (int) probs.size(); i++) {
//    for (int i = 0; i < 10; i++) {
//        printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
//    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;

}
