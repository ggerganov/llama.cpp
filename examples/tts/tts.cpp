#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "tts-impl.hpp"

#define _USE_MATH_DEFINES // For M_PI on MSVC

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

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

static void save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for writing", __func__, fname.c_str());
        return;
    }

    wav_header header;
    header.sample_rate = sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = data.size() * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto & sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    file.close();
}

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "";

    params.n_predict = 4096;
    params.n_batch   = 8192;
    params.n_ctx     = 8192;

    params.sampling.top_k = 4;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    const int n_parallel = params.n_parallel;
    const int n_predict  = params.n_predict;

    common_init();

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_ttc = NULL; // text-to-codes
    llama_model * model_cts = NULL; // codes-to-speech

    llama_context * ctx_ttc = NULL;
    llama_context * ctx_cts = NULL;

    common_init_result llama_init_ttc = common_init_from_params(params);

    model_ttc = llama_init_ttc.model.get();
    ctx_ttc   = llama_init_ttc.context.get();

    // TODO: refactor in a common struct
    params.model     = params.vocoder.model;
    params.model_url = params.vocoder.model_url;
    params.hf_repo   = params.vocoder.hf_repo;
    params.hf_file   = params.vocoder.hf_file;

    params.embedding = true;

    common_init_result llama_init_cts = common_init_from_params(params);

    model_cts = llama_init_cts.model.get();
    ctx_cts   = llama_init_cts.context.get();

    std::vector<common_sampler *> smpl(n_parallel);
    for (int i = 0; i < n_parallel; ++i) {
        params.sampling.no_perf = (i != 0);
        params.sampling.seed = params.sampling.seed + 1;

        smpl[i] = common_sampler_init(model_ttc, params.sampling);
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl[0]));
    LOG_INF("sampler params: \n%s\n", params.sampling.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl[0]).c_str());

    LOG_INF("%s: loading done\n", __func__);

    const auto t_main_start = ggml_time_us();

    std::vector<llama_token> codes;

    // process prompt and generate voice codes
    {
        LOG_INF("%s: constructing prompt ..\n", __func__);

        std::vector<llama_token> prompt_inp = tts_preprocess_prompt(model_ttc, params.prompt);

        // print the prompt token-by-token

        LOG("\n");

        for (auto id : prompt_inp) {
            LOG("%s", common_token_to_piece(ctx_ttc, id).c_str());
        }

        LOG_INF("%s: prompt size: %d\n", __func__, (int) prompt_inp.size());

        LOG("\n");

        // create a llama_batch
        // we use this object to submit token data for decoding
        llama_batch batch = llama_batch_init(std::max(prompt_inp.size(), (size_t) n_parallel), 0, n_parallel);

        std::vector<llama_seq_id> seq_ids(n_parallel, 0);
        for (int32_t i = 0; i < n_parallel; ++i) {
            seq_ids[i] = i;
        }

        // evaluate the initial prompt
        for (size_t i = 0; i < prompt_inp.size(); ++i) {
            common_batch_add(batch, prompt_inp[i], i, seq_ids, false);
        }
        GGML_ASSERT(batch.n_tokens == (int) prompt_inp.size());

        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx_ttc, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        if (n_parallel > 1) {
            LOG_INF("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
        }

        llama_synchronize(ctx_ttc);

        LOG_INF("%s: time for prompt: %.3f ms\n\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

        const auto t_dec_start = ggml_time_us();

        // main loop

        // remember the batch index of the last token for each parallel sequence
        // we need this to determine which logits to sample from
        std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

        int n_past   = batch.n_tokens;
        int n_decode = 0;

        while (n_decode <= n_predict) {
            // prepare the next batch
            common_batch_clear(batch);

            // sample the next token for each parallel sequence / stream
            for (int32_t i = 0; i < n_parallel; ++i) {
                if (i_batch[i] < 0) {
                    // the stream has already finished
                    continue;
                }

                const llama_token new_token_id = common_sampler_sample(smpl[i], ctx_ttc, i_batch[i]);

                common_sampler_accept(smpl[i], new_token_id, true);

                codes.push_back(new_token_id);

                const auto * cands = common_sampler_get_candidates(smpl[i]);

                // is it an end of generation? -> mark the stream as finished
                if (llama_token_is_eog(model_ttc, new_token_id) || n_decode == n_predict) {
                    std::string reason;
                    if (llama_token_is_eog(model_ttc, new_token_id)) {
                        reason = "eos";
                    } else {
                        reason = "n_predict";
                    }

                    i_batch[i] = -1;

                    LOG("\n");
                    if (n_parallel > 1) {
                        LOG_CNT("\n");
                        LOG_INF("%s: stream %d finished at n_past = %d, reason = '%s'\n", __func__, i, n_past, reason.c_str());
                    }

                    continue;
                }

                {
                    const float p = cands->data[cands->selected].p;

                    const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) ((3*p)*float(k_colors.size()))));

                    LOG_CNT("%s%d%s", k_colors[col].c_str(), i, "\033[0m");
                    //LOG_CNT("%d", i);
                }

                i_batch[i] = batch.n_tokens;

                // push this new token for next evaluation
                common_batch_add(batch, new_token_id, n_past, { i }, true);
            }

            // all streams are finished
            if (batch.n_tokens == 0) {
                break;
            }

            n_decode += 1;
            n_past += 1;

            // evaluate the current batch with the transformer model
            if (llama_decode(ctx_ttc, batch)) {
                LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }
        }

        llama_batch_free(batch);

        LOG("\n");
        LOG_INF("%s: time for decoder:       %.3f ms\n", __func__, (ggml_time_us() - t_dec_start) / 1000.0f);
    }

    common_perf_print(ctx_ttc, smpl[0]);

    //std::vector<llama_token> codes = {198, 88225, 155856, 151669, 152205,
    //    153064, 152537, 153421, 153209, 152524, 151689, 152993, 152438, 152695,
    //    153091, 152945, 152829, 152534, 152934, 153020, 151997, 152263, 153010,
    //    153146, 152399, 153208, 152496, 151793, 152848, 152263, 152571, 153286,
    //    152227, 153300, 152934, 152263, 153208, 152263, 152965, 152430, 152296,
    //    153146, 152920, 152376, 152556, 153363, 151775, 152044, 152972, 152690,
    //    153379, 152368, 152233, 153422, 152490, 151996, 152022, 151694, 152061,
    //    153238, 152539, 153356, 152640, 153021, 153123, 151962, 153094, 151670,
    //    198, 20339, 13189, 155824, 151669, 152070, 152007, 152910, 151683,
    //    152000, 152373, 152760, 152046, 151735, 152334, 152394, 153073, 152908,
    //    151856, 151953, 153247, 153293, 151903, 153480, 153168, 152478, 153359,
    //    153429, 151905, 151678, 152567, 152411, 152165, 152556, 153075, 153424,
    //    151993, 152999, 153078, 152151, 152088, 153389, 152484, 151874, 151670,
    //    198, 285, 155784, 151669, 152226, 152126, 152638, 153215, 151729,
    //    152959, 153479, 153059, 151838, 151670, 198, 1782, 155783, 151669,
    //    153288, 153055, 153314, 152497, 152962, 152741, 152076, 153253, 151670,
    //    198, 471, 16488, 155825, 151669, 152060, 152916, 151893, 153469, 152501,
    //    152080, 152743, 151932, 153161, 152096, 152761, 152698, 153401, 153242,
    //    153336, 152441, 152838, 153467, 152706, 153496, 153310, 152422, 153360,
    //    153115, 152763, 151998, 152373, 153450, 152554, 151968, 153323, 152055,
    //    152468, 153111, 153358, 152813, 152010, 151770, 152823, 152960, 151670,
    //    198, 22627, 155823, 151669, 152814, 152366, 153484, 152931, 153441,
    //    152164, 152877, 152915, 153463, 151692, 152911, 152747, 152776, 151831,
    //    153449, 151882, 152975, 152031, 152513, 153150, 152448, 152667, 153133,
    //    153189, 152619, 153466, 152054, 152106, 153119, 152277, 152439, 153109,
    //    152997, 152141, 153154, 153256, 153311, 151922, 151670, 198, 1055,
    //    155781, 151669, 152633, 151850, 153060, 153270, 152560, 153348, 152729,
    //    151670, 198, 25312, 155803, 151669, 152521, 153403, 152561, 153337,
    //    153383, 152199, 153493, 153326, 151830, 152254, 152248, 152349, 152153,
    //    153007, 151823, 153037, 152575, 152457, 152406, 152592, 153116, 153365,
    //    153456, 151670, 198, 88225, 155817, 151669, 153271, 151925, 152218,
    //    152418, 152253, 153140, 151903, 153151, 152626, 152338, 152647, 153464,
    //    152785, 152768, 151711, 152037, 152033, 151804, 152216, 151701, 151855,
    //    152348, 152995, 152955, 152905, 152342, 152340, 153391, 153453, 152418,
    //    153415, 151990, 153083, 152884, 151670, 198, 151668, 198, 151645};

    {
        const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);

        LOG("\n");
        LOG_INF("codes: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: codes size: %d\n", __func__, (int) codes.size());
    }

    // remove all non-audio tokens (i.e. < 151672 || > 155772)
    codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());

    {
        const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);
        LOG_INF("codes audio: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: codes audio size: %d\n", __func__, (int) codes.size());
    }

    const auto t_voc_start = ggml_time_us();

    std::vector<float> embd;
    if (tts_get_embd(ctx_cts, codes, embd) != 0) {
        LOG_ERR("%s: tts_get_embd() failed\n", __func__);
        return 1;
    }

    LOG_INF("%s: time for vocoder:      %.3f ms\n", __func__, (ggml_time_us() - t_voc_start) / 1000.0f);

    const auto t_spec_start = ggml_time_us();

#if 1
    // spectral operations
    const int n_embd = llama_n_embd(model_cts);
    const int n_codes = codes.size();

    auto audio = tts_embd_to_audio(embd.data(), n_codes, n_embd, params.cpuparams.n_threads);
#else
    // read the spectrogram from a file for debugging purposes
    std::vector<float> audio;
    {
        std::ifstream fin("out.bin", std::ios::binary);
        if (!fin) {
            LOG_ERR("%s: failed to open file '%s'\n", __func__, "out.bin");
            return 1;
        }

        std::vector<float> embd;

        int n_codes;
        int n_embd;

        fin.read(reinterpret_cast<char *>(&n_codes), sizeof(int));
        fin.read(reinterpret_cast<char *>(&n_embd), sizeof(int));

        embd.resize(n_codes * n_embd);
        fin.read(reinterpret_cast<char *>(embd.data()), n_codes * n_embd * sizeof(float));
        fin.close();

        LOG_INF("%s: n_codes: %d, n_embd: %d\n", __func__, n_codes, n_embd);

        audio = embd_to_audio(embd.data(), n_codes, n_embd, params.cpuparams.n_threads);
    }
#endif

    const std::string fname = "output.wav";

    const int n_sr = 24000; // sampling rate

    // zero out first 0.25 seconds
    for (int i = 0; i < 24000/4; ++i) {
        audio[i] = 0.0f;
    }

    LOG_INF("%s: time for spectral ops: %.3f ms\n", __func__, (ggml_time_us() - t_spec_start) / 1000.0f);
    LOG_INF("%s: total time:            %.3f ms\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

    save_wav16(fname, audio, n_sr);

    LOG_INF("%s: audio written to file '%s'\n", __func__, fname.c_str());

    llama_backend_free();

    return 0;
}
