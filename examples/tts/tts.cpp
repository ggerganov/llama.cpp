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
#include <thread>

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

static void fill_hann_window(int length, bool periodic, double * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

// very poor-man fft
static void twiddle(double * real, double * imag, int k, int N) {
    double angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

static void irfft(int n, const double * inp_cplx, double * out_real) {
    int N = n / 2 + 1;

    std::vector<double> real_input(N);
    std::vector<double> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<double> real_output(n);
    std::vector<double> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            double twiddle_real;
            double twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

//
//  y = torch.nn.functional.fold(
//       data, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
//  )[:, 0, 0, pad:-pad]
//
// data.shape =  torch.Size([1, 1280, 261])
// output_size =  84480
// win_length =  1280
// hop_length =  320
// pad =  480
//
static void fold(const std::vector<double> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<double> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

static void save_wav16(const std::string & fname, const std::vector<double> & data, int sample_rate) {
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

static std::vector<double> embd_to_audio(
        const float * embd,
        const std::vector<llama_token> & codes,
        const int n_embd,
        const int n_thread) {
    const int n     = codes.size();
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n - 1)*n_hop + n_win;

    std::vector<double> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n;

    std::vector<double> E (n_spec);
    std::vector<double> S (n_spec);
    std::vector<double> ST(n_spec);

    for (int l = 0; l < n; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n; ++l) {
            double mag = E[(k           )*n + l];
            double phi = E[(k + n_embd/2)*n + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n + l) + 0] = mag*cosf(phi);
            S[2*(k*n + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n + l) + 1];
        }
    }

    std::vector<double> res  (n*n_fft);
    std::vector<double> hann2(n*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<double> audio;
    std::vector<double> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
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

    std::vector<llama_token> codes = {198, 88225, 155856, 151669, 152205,
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
        const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);
        LOG_INF("prompt: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: prompt size: %d\n", __func__, (int) codes.size());
    }

    // remove all non-audio tokens (i.e. < 151672 || > 155772)
    codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());

    {
        const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);
        LOG_INF("prompt audio: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: prompt audio size: %d\n", __func__, (int) codes.size());
    }

    for (auto & token : codes) {
        token -= 151672;
    }

    const auto t_voc_start = ggml_time_us();

    llama_batch batch = llama_batch_init(codes.size(), 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < codes.size(); ++i) {
        common_batch_add(batch, codes[i], i, { 0 }, true); // TODO: all logits?
    }
    GGML_ASSERT(batch.n_tokens == (int) codes.size());

    if (llama_decode(ctx_cts, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    llama_synchronize(ctx_cts);

    LOG_INF("%s: time for vocoder:      %.3f ms\n", __func__, (ggml_time_us() - t_voc_start) / 1000.0f);

    const auto t_spec_start = ggml_time_us();

    const int n_embd = llama_n_embd(model_cts);
    const float * embd = llama_get_embeddings(ctx_cts);

    // spectral operations
    // TODO: not optimized at all
    auto audio = embd_to_audio(embd, codes, n_embd, params.cpuparams.n_threads);

    const std::string fname = "output.wav";

    const int n_sr = 24000; // sampling rate

    LOG_INF("%s: time for spectral ops: %.3f ms\n", __func__, (ggml_time_us() - t_spec_start) / 1000.0f);
    LOG_INF("%s: total time:            %.3f ms\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

    save_wav16(fname, audio, n_sr);

    LOG_INF("%s: audio written to file '%s'\n", __func__, fname.c_str());

    llama_free(ctx_ttc);
    llama_free_model(model_ttc);

    llama_free(ctx_cts);
    llama_free_model(model_cts);

    llama_backend_free();

    return 0;
}
