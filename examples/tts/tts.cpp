#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

//
// Terminal utils
//

#define SQR(X)    ((X) * (X))
#define UNCUBE(x) x < 48 ? 0 : x < 115 ? 1 : (x - 35) / 40

/**
 * Quantizes 24-bit RGB to xterm256 code range [16,256).
 */
static int rgb2xterm256(int r, int g, int b) {
    unsigned char cube[] = {0, 0137, 0207, 0257, 0327, 0377};
    int av, ir, ig, ib, il, qr, qg, qb, ql;
    av = r * .299 + g * .587 + b * .114 + .5;
    ql = (il = av > 238 ? 23 : (av - 3) / 10) * 10 + 8;
    qr = cube[(ir = UNCUBE(r))];
    qg = cube[(ig = UNCUBE(g))];
    qb = cube[(ib = UNCUBE(b))];
    if (SQR(qr - r) + SQR(qg - g) + SQR(qb - b) <=
        SQR(ql - r) + SQR(ql - g) + SQR(ql - b))
        return ir * 36 + ig * 6 + ib + 020;
    return il + 0350;
}

static std::string set_xterm256_foreground(int r, int g, int b) {
    int x = rgb2xterm256(r, g, b);
    std::ostringstream oss;
    oss << "\033[38;5;" << x << "m";
    return oss.str();
}

const std::vector<std::string> k_colors = {
    set_xterm256_foreground(220,   5,  12),
    set_xterm256_foreground(232,  96,  28),
    set_xterm256_foreground(241, 147,  45),
    set_xterm256_foreground(246, 193,  65),
    set_xterm256_foreground(247, 240,  86),
    set_xterm256_foreground(144, 201, 135),
    set_xterm256_foreground( 78, 178, 101),
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"Hello!\"\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "";

    params.n_predict = 1024;
    params.n_batch   = 8192;
    params.n_ctx     = 8192;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    common_init();

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_ttc = NULL; // text-to-codes
    llama_model * model_cts = NULL; // codes-to-speech

    llama_context * ctx_ttc = NULL;
    llama_context * ctx_cts = NULL;

    common_init_result llama_init_ttc = common_init_from_params(params);
    model_ttc = llama_init_ttc.model;
    ctx_ttc = llama_init_ttc.context;

    params.model = params.vocoder.model;
    params.embedding = true;

    common_init_result llama_init_cts = common_init_from_params(params);
    model_cts = llama_init_cts.model;
    ctx_cts = llama_init_cts.context;

    const auto t_main_start = ggml_time_us();

    std::vector<llama_token> prompt_inp = {198, 88225, 155856, 151669, 152205,
        153064, 152537, 153421, 153209, 152524, 151689, 152993, 152438, 152695,
        153091, 152945, 152829, 152534, 152934, 153020, 151997, 152263, 153010,
        153146, 152399, 153208, 152496, 151793, 152848, 152263, 152571, 153286,
        152227, 153300, 152934, 152263, 153208, 152263, 152965, 152430, 152296,
        153146, 152920, 152376, 152556, 153363, 151775, 152044, 152972, 152690,
        153379, 152368, 152233, 153422, 152490, 151996, 152022, 151694, 152061,
        153238, 152539, 153356, 152640, 153021, 153123, 151962, 153094, 151670,
        198, 20339, 13189, 155824, 151669, 152070, 152007, 152910, 151683,
        152000, 152373, 152760, 152046, 151735, 152334, 152394, 153073, 152908,
        151856, 151953, 153247, 153293, 151903, 153480, 153168, 152478, 153359,
        153429, 151905, 151678, 152567, 152411, 152165, 152556, 153075, 153424,
        151993, 152999, 153078, 152151, 152088, 153389, 152484, 151874, 151670,
        198, 285, 155784, 151669, 152226, 152126, 152638, 153215, 151729,
        152959, 153479, 153059, 151838, 151670, 198, 1782, 155783, 151669,
        153288, 153055, 153314, 152497, 152962, 152741, 152076, 153253, 151670,
        198, 471, 16488, 155825, 151669, 152060, 152916, 151893, 153469, 152501,
        152080, 152743, 151932, 153161, 152096, 152761, 152698, 153401, 153242,
        153336, 152441, 152838, 153467, 152706, 153496, 153310, 152422, 153360,
        153115, 152763, 151998, 152373, 153450, 152554, 151968, 153323, 152055,
        152468, 153111, 153358, 152813, 152010, 151770, 152823, 152960, 151670,
        198, 22627, 155823, 151669, 152814, 152366, 153484, 152931, 153441,
        152164, 152877, 152915, 153463, 151692, 152911, 152747, 152776, 151831,
        153449, 151882, 152975, 152031, 152513, 153150, 152448, 152667, 153133,
        153189, 152619, 153466, 152054, 152106, 153119, 152277, 152439, 153109,
        152997, 152141, 153154, 153256, 153311, 151922, 151670, 198, 1055,
        155781, 151669, 152633, 151850, 153060, 153270, 152560, 153348, 152729,
        151670, 198, 25312, 155803, 151669, 152521, 153403, 152561, 153337,
        153383, 152199, 153493, 153326, 151830, 152254, 152248, 152349, 152153,
        153007, 151823, 153037, 152575, 152457, 152406, 152592, 153116, 153365,
        153456, 151670, 198, 88225, 155817, 151669, 153271, 151925, 152218,
        152418, 152253, 153140, 151903, 153151, 152626, 152338, 152647, 153464,
        152785, 152768, 151711, 152037, 152033, 151804, 152216, 151701, 151855,
        152348, 152995, 152955, 152905, 152342, 152340, 153391, 153453, 152418,
        153415, 151990, 153083, 152884, 151670, 198, 151668, 198, 151645};

    {
        const std::string inp_txt = common_detokenize(ctx_ttc, prompt_inp, true);
        LOG_INF("prompt: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: prompt size: %d\n", __func__, (int) prompt_inp.size());
    }

    // remove all non-audio tokens (i.e. < 151672 || > 155772)
    prompt_inp.erase(std::remove_if(prompt_inp.begin(), prompt_inp.end(), [](llama_token t) { return t < 151672 || t > 155772; }), prompt_inp.end());

    {
        const std::string inp_txt = common_detokenize(ctx_ttc, prompt_inp, true);
        LOG_INF("prompt audio: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: prompt audio size: %d\n", __func__, (int) prompt_inp.size());
    }

    for (auto & token : prompt_inp) {
        token -= 151672;
    }

    llama_batch batch = llama_batch_init(prompt_inp.size(), 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < prompt_inp.size(); ++i) {
        common_batch_add(batch, prompt_inp[i], i, { 0 }, true); // TODO: all logits?
    }
    GGML_ASSERT(batch.n_tokens == (int) prompt_inp.size());

    if (llama_decode(ctx_cts, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    llama_synchronize(ctx_cts);

    LOG_INF("%s: time for prompt: %.3f ms\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

    const float * embd = llama_get_embeddings(ctx_cts);

    int n = 768*261;

    LOG("result:\n");
    for (int i = 0; i < 10; ++i) {
        LOG("%8.3f ", embd[i]);
    }
    LOG("\n");
    for (int i = n - 10; i < n; ++i) {
        LOG("%8.3f ", embd[i]);
    }
    LOG("\n");
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += embd[i];
    }
    LOG("sum: %f\n", sum);

    fprintf(stderr, "\n");

    llama_free(ctx_ttc);
    llama_free_model(model_ttc);

    llama_free(ctx_cts);
    llama_free_model(model_cts);

    llama_backend_free();

    return 0;
}
