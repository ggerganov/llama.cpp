#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <bitset>
#include <fstream>

int msB_log256(int x)
{
    int ret = 0;
    while (x > 0)
    {
        ret++;
        x >>= 8;
    }
    return ret;
}

const int block_header_size = 2;
const int fixed_token_cost = 1;

int total_pad = 0;

std::vector<uint8_t> encode(llama_context *ctx, std::vector<llama_token> inp, gpt_sampler *smpl, int num_raw_tokens_header)
{

    llama_batch batch = llama_batch_init(inp.size(), 0, 1);

    for (size_t i = 0; i < num_raw_tokens_header; i++)
    {
        llama_batch_add(batch, inp[i], i, {0}, true);
    }

    // eval the first few tokens of the prompt
    if (llama_decode(ctx, batch))
    {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        exit(1);
    }

    const auto t_enc_end = ggml_time_us();

    std::vector<int> sample_ids;

    gpt_sampler_sample(smpl, ctx, num_raw_tokens_header - 1, true);
    for (int index = num_raw_tokens_header; index < inp.size(); index++)
    {
        auto cur_p = gpt_sampler_get_candidates(smpl); // initialized by set_logits

        int match = -1;
        for (int i = 0; i < cur_p->size; i++)
        {
            auto tok = cur_p->data[i];
            llama_token candidate = tok.id;
            if (candidate == inp[index])
            {
                LOG("%s", llama_token_to_piece(ctx, candidate).c_str());
                match = i;
                break;
            }
        }
        if (match < 0)
        {
            LOG_ERR("\n couldn't match %s", llama_token_to_piece(ctx, inp[index]).c_str());
            exit(1);
        }
        sample_ids.push_back(match);
        llama_batch_clear(batch);
        llama_batch_add(batch, inp[index], index, {0}, true);
        if (llama_decode(ctx, batch))
        {
            LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
            exit(1);
        }
        gpt_sampler_sample(smpl, ctx, 0, true);
    }

    std::vector<uint8_t> sample_ids_bitpacked;

    int bit_offset = 0;
    uint8_t current = 0;

    int block_start = 0;
    bool build_block = true;
    bool was_block = false;

    // frst put the raw first few tokens
    sample_ids_bitpacked.push_back(num_raw_tokens_header);
    for (size_t i = 0; i < num_raw_tokens_header; i++)
    {
        // pack 4 bytes
        for (int j = 0; j < 4; j++)
        {
            uint8_t byte = inp[i] >> (j * 8);
            sample_ids_bitpacked.push_back(byte);
        }
    }
    block_start = 1 + num_raw_tokens_header * 4;
    bit_offset = block_start * 8;

    for (int i = 0; i < sample_ids.size(); i++)
    {
        int sample_id = sample_ids[i];
        uint8_t PAD = (8 - bit_offset % 8) % 8;
        uint8_t bytesize = (uint8_t)msB_log256(sample_id);

        // Big number, better save as token
        if (sample_id > PAD + (block_header_size + fixed_token_cost + bytesize) * 8)
        {
            // Close current block (0b1010 is block marker)
            if (was_block)
            {
                sample_ids_bitpacked[block_start] = 0b10100000 | PAD;
                int block_size = (bit_offset + PAD) / 8 - block_start;
                if (block_size >= 256)
                {
                    // TODO: handle more than 256 bytes of block data
                    // (maybe allow multiple blocks in a row)
                    LOG_ERR("Block too big %d >= 256", block_size);
                    exit(-1);
                }
                sample_ids_bitpacked[block_start + 1] = block_size & 0xff;

                // put last bytes
                if (PAD)
                {
                    sample_ids_bitpacked.push_back(current);
                    current = 0;
                }
            }
            bit_offset += PAD;
            total_pad += PAD;
            if (bit_offset % 8)
            {
                LOG_ERR("Unreachable");
                exit(-1);
            }
            // 0b0101 is token marker
            sample_ids_bitpacked.push_back(0b01010000 | bytesize);
            // put token bytes into sample_ids_bitpacked
            for (int j = 0; j < bytesize; j++)
            {
                sample_ids_bitpacked.push_back(sample_id & 0xff);
                sample_id >>= 8;
            }
            if (sample_id)
                LOG("Shouldn't happen");
            bit_offset += 8 * (fixed_token_cost + bytesize);
            build_block = true;
            was_block = false;
            continue;
        }
        was_block = true;
        if (build_block)
        {
            if (bit_offset % 8)
            {
                LOG_ERR("Unreachable");
                exit(-1);
            }
            build_block = false;
            block_start = bit_offset / 8;
            for (int j = 0; j < block_header_size; j++)
            {
                sample_ids_bitpacked.push_back(0);
            }
            bit_offset += 8 * block_header_size;
        }
        for (int j = 0; j < sample_id; j++)
        {
            current |= 1 << (7 - bit_offset % 8);
            bit_offset++;
            if (bit_offset % 8 == 0)
            {
                sample_ids_bitpacked.push_back(current);
                current = 0;
            }
        }
        bit_offset++;
        if (bit_offset % 8 == 0)
        {
            sample_ids_bitpacked.push_back(current);
            current = 0;
        }
    }
    if (!build_block)
    {
        if (bit_offset % 8)
            sample_ids_bitpacked.push_back(current);
        uint8_t PAD = (8 - bit_offset % 8) % 8;
        sample_ids_bitpacked[block_start] = 0b10100000 | PAD;
        int block_size = (bit_offset + PAD) / 8 - block_start;
        // endianness: big endian
        sample_ids_bitpacked[block_start + 1] = block_size & 0xff;
        total_pad += PAD;
    }
    llama_batch_free(batch);
    return sample_ids_bitpacked;
}

std::vector<llama_token> decode(llama_context *ctx, gpt_sampler *smpl, std::vector<uint8_t> sample_ids_bitpacked, std::vector<llama_token> inp = {})
{
    std::vector<llama_token> out;

    llama_batch batch = llama_batch_init(512, 0, 1);

    int num_raw_tokens_header = sample_ids_bitpacked[0];

    for (size_t i = 0; i < num_raw_tokens_header; i++)
    {
        // unpack 4 bytes
        llama_token token = 0;
        for (int j = 3; j >= 0; j--)
        {
            token <<= 8;
            token |= sample_ids_bitpacked[1 + i * 4 + j];
        }

        llama_batch_add(batch, token, i, {0}, true);
        out.push_back(token);
        auto token_str = llama_token_to_piece(ctx, token);
        LOG("%s", token_str.c_str());
    }
    if (llama_decode(ctx, batch))
    {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        exit(1);
    }

    gpt_sampler_sample(smpl, ctx, num_raw_tokens_header - 1, true);

    int index = 0;
    int bit_index = (1 + num_raw_tokens_header * 4) * 8;
    const int bitsize = sample_ids_bitpacked.size() * 8;
    while (bit_index < bitsize)
    {

        uint8_t header = sample_ids_bitpacked[bit_index / 8];
        if (header & 0b01010000)
        {
            uint8_t bytesize = header & 0x0f;
            // it's a token

            int sample_id = 0;
            for (int i = bytesize; i > 0; i--)
            {
                sample_id <<= 8;
                sample_id |= (int)sample_ids_bitpacked[i + (bit_index / 8)];
            }

            auto cur_p = gpt_sampler_get_candidates(smpl); // initialized by set_logits

            auto token_id = cur_p->data[sample_id].id;

            out.push_back(token_id);

            if (!inp.size() || token_id == inp[num_raw_tokens_header + index])
            {
                LOG("%s", llama_token_to_piece(ctx, token_id).c_str());
            }
            else
            {
                // print in red
                LOG("\u001b[31m%s", llama_token_to_piece(ctx, token_id).c_str());
                LOG("\nExpected: %s", llama_token_to_piece(ctx, inp[num_raw_tokens_header + index]).c_str());
                LOG("\n, Id: %d != %d", token_id, inp[num_raw_tokens_header + index]);
                LOG("\nPos: %d, bs:%d", sample_id, bytesize);

                // print sample_id bytes in hex
                LOG("\n");
                for (int i = bytesize; i > 0; i--)
                {
                    LOG("%02x ", sample_ids_bitpacked[i + (bit_index / 8)]);
                }
                exit(-1);
            }

            llama_batch_clear(batch);
            llama_batch_add(batch, token_id, num_raw_tokens_header + index, {0}, true);
            if (llama_decode(ctx, batch))
            {
                LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
                exit(1);
            }
            gpt_sampler_sample(smpl, ctx, 0, true);

            index++;

            bit_index += 8 * (fixed_token_cost + bytesize);
        }
        else
        {
            // it's a block
            uint8_t PAD = header & 0x0f;
            int block_size = sample_ids_bitpacked[bit_index / 8 + 1];
            int block_end = block_size * 8 + bit_index;
            bit_index += 8 * block_header_size;
            int id = 0;
            for (; bit_index < block_end - PAD; bit_index++)
            {
                bool bit = sample_ids_bitpacked[bit_index / 8] & (1 << (7 - bit_index % 8));
                if (bit)
                {
                    id++;
                }
                else
                {
                    {
                        int sample_id = id;

                        auto cur_p = gpt_sampler_get_candidates(smpl); // initialized by set_logits

                        auto token_id = cur_p->data[sample_id].id;
                        out.push_back(token_id);
                        if (!inp.size() || token_id == inp[num_raw_tokens_header + index])
                        {
                            LOG("%s", llama_token_to_piece(ctx, token_id).c_str());
                        }
                        else
                        {
                            // print in red
                            LOG("\u001b[31m%s", llama_token_to_piece(ctx, token_id).c_str());
                        }

                        llama_batch_clear(batch);
                        llama_batch_add(batch, token_id, num_raw_tokens_header + index, {0}, true);
                        if (llama_decode(ctx, batch))
                        {
                            LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
                            exit(1);
                        }
                        gpt_sampler_sample(smpl, ctx, 0, true);
                    }
                    index++;

                    id = 0;
                }
            }
            bit_index += PAD;
        }
    }

    llama_batch_free(batch);
    return out;
}

void test(const gpt_params &params)
{
    int num_raw_tokens_header = params.num_tokens_header;
    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    // Tokenize the prompt
    std::vector<llama_token> inp;

    inp = ::llama_tokenize(ctx, params.prompt, false, false);

    // num_raw_tokens_header = inp.size();
    assert(inp.size() > num_raw_tokens_header);

    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int)inp.size() > max_tokens_list_size)
    {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int)inp.size(), max_tokens_list_size);
        exit(1);
    }

    LOG("\n\n");

    int i = 0;
    for (auto id : inp)
    {
        LOG("%s", llama_token_to_piece(ctx, id).c_str());
        if (++i >= num_raw_tokens_header)
            break;
    }

    fflush(stderr);

    // encode stage

    const auto t_enc_start = ggml_time_us();

    struct gpt_sampler *smpl = gpt_sampler_init(model, params.sparams);

    std::vector<uint8_t> sample_ids_bitpacked = encode(ctx, inp, smpl, num_raw_tokens_header);

    gpt_sampler_free(smpl);
    auto t_enc_end = ggml_time_us();

    LOG("\n");

    // print bits as binary to debug
    for (int i = 0; i < sample_ids_bitpacked.size(); i++)
    {
        std::bitset<8> x(sample_ids_bitpacked[i]);
        LOG("%s ", x.to_string().c_str());
    }
    LOG("\n");

    // print as hexadecimal
    for (int i = 0; i < sample_ids_bitpacked.size(); i++)
    {
        LOG("%02X ", sample_ids_bitpacked[i]);
    }
    LOG("\n");

    LOG("\nInput: %d characters (%d tokens)", params.prompt.length(), inp.size());

    float compressed_byte_per_token = (float)sample_ids_bitpacked.size() / (float)inp.size();
    float compressed_bits_per_char = 8 * (float)sample_ids_bitpacked.size() / (float)params.prompt.length();

    LOG("\n%d compressed bytes,(%04f bytes per token, %04f bits per character)\n", (int)sample_ids_bitpacked.size(), compressed_byte_per_token, compressed_bits_per_char);

    llama_free(ctx);
    ctx = llama_new_context_with_model(model, ctx_params);

    LOG("\n------------\n");

    // decode stage

    const auto t_dec_start = ggml_time_us();

    smpl = gpt_sampler_init(model, params.sparams);
    decode(ctx, smpl, sample_ids_bitpacked, inp);

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", inp.size(), (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("\n");

    LOG_INF("\n");
    gpt_perf_print(ctx, smpl);

    gpt_sampler_free(smpl);

    llama_free(ctx);
    llama_free_model(model);
}

int main(int argc, char **argv)
{
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPRESS))
    {
        return 1;
    }

    // TODO: change defaults instead?
    params.sparams.min_p = 0;
    params.sparams.top_p = 1;
    params.sparams.top_k = -1;
    // Avoid temp=0 because greedy sampling breaks stuff
    params.sparams.temp = 1.;

    gpt_init();

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: use Enum?
    if (params.compress_mode == 0)
    {
        test(params);
    }
    else if (params.compress_mode == 1)
    { // compress
        llama_model_params model_params = llama_model_params_from_gpt_params(params);
        llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

        llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
        llama_context *ctx = llama_new_context_with_model(model, ctx_params);

        // Tokenize the prompt
        std::vector<llama_token> inp;

        inp = ::llama_tokenize(ctx, params.prompt, false, false);

        assert(inp.size() > params.num_tokens_header);

        const int max_context_size = llama_n_ctx(ctx);
        const int max_tokens_list_size = max_context_size - 4;

        if ((int)inp.size() > max_tokens_list_size)
        {
            LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int)inp.size(), max_tokens_list_size);
            return 1;
        }

        // Eval the start of the prompt
        int i = 0;
        for (auto id : inp)
        {
            LOG("%s", llama_token_to_piece(ctx, id).c_str());
            if (++i >= params.num_tokens_header)
                break;
        }

        fflush(stderr);

        // encode stage

        const auto t_enc_start = ggml_time_us();

        struct gpt_sampler *smpl = gpt_sampler_init(model, params.sparams);

        std::vector<uint8_t> sample_ids_bitpacked = encode(ctx, inp, smpl, params.num_tokens_header);

        gpt_sampler_free(smpl);
        llama_free(ctx);
        llama_free_model(model);
        auto t_enc_end = ggml_time_us();

        LOG("\n");
        if (!params.no_perf)
        {
            LOG("\nInput: %d characters (%d tokens)", params.prompt.length(), inp.size());

            float compressed_bits_per_token = 8 * (float)sample_ids_bitpacked.size() / (float)inp.size();
            float compressed_bits_per_char = 8 * (float)sample_ids_bitpacked.size() / (float)params.prompt.length();

            LOG("\n%d compressed bytes,(%04f bits per token, %04f bits per character)\n", (int)sample_ids_bitpacked.size(), compressed_bits_per_token, compressed_bits_per_char);
            LOG("\n%d padding bits, (%04f bits per character without padding)", total_pad, compressed_bits_per_char - total_pad / (float)params.prompt.length());
            LOG("\nPPL (over)estimation: %04f (%04f with padding)", exp2(compressed_bits_per_token - total_pad / (float)inp.size()), exp2(compressed_bits_per_token));
        }
        // maybe this needs to be changed
        if (params.out_file != "imatrix.dat")
        {
            // dump uint8array to bin file
            std::ofstream ofs(params.out_file.c_str(), std::ios::binary);
            ofs.write((char *)&sample_ids_bitpacked[0], sample_ids_bitpacked.size());
            ofs.close();
        }
        else
        {
            LOG("\n------------\n");
            // print as hex to stdout
            for (int i = 0; i < sample_ids_bitpacked.size(); i++)
            {
                LOG("%02X ", sample_ids_bitpacked[i]);
            }
        }
    }
    else if (params.compress_mode == 2)
    {
        // decompress mode
        //  load sample_ids_bitpacked from params.prompt_file
        std::ifstream ifs(params.prompt_file.c_str(), std::ios::binary);

        if (!ifs)
        {
            LOG_ERR("%s: failed to open file\n", __func__);
            return -1;
        }
        // Get the ifs size
        ifs.seekg(0, std::ios::end);
        std::streampos fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        // Reserve space in the vector
        std::vector<uint8_t> sample_ids_bitpacked(fileSize);

        // Read the ifs into the vector
        if (!ifs.read(reinterpret_cast<char *>(sample_ids_bitpacked.data()), fileSize))
        {
            LOG_ERR("%s: failed to read file\n", __func__);
            return -1;
        }
        ifs.close();

        // Debug: print as hex
        for (int i = 0; i < sample_ids_bitpacked.size(); i++)
        {
            LOG("%02X ", sample_ids_bitpacked[i]);
        }
        LOG("\n");

        llama_model_params model_params = llama_model_params_from_gpt_params(params);
        llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

        llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
        llama_context *ctx = llama_new_context_with_model(model, ctx_params);

        const auto t_dec_start = ggml_time_us();

        struct gpt_sampler *smpl = gpt_sampler_init(model, params.sparams);

        std::vector<llama_token> out = decode(ctx, smpl, sample_ids_bitpacked);

        gpt_sampler_free(smpl);
        auto t_dec_end = ggml_time_us();

        // maybe this needs to be changed
        if (params.out_file != "imatrix.dat")
        {
            // dump as string to file
            std::string out_str = ::llama_detokenize(ctx, out);

            std::ofstream ofs(params.out_file.c_str(), std::ios::binary);
            ofs.write((char *)&out_str[0], out_str.size());
            ofs.close();
        }

        llama_free(ctx);
        llama_free_model(model);
    }

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
