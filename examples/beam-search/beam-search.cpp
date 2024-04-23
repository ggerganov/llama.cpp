#include "common.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// Used for debugging to print out beam tokens.
struct ostream_beam_view {
    llama_context * ctx;
    llama_beam_view beam_view;
};

static std::ostream & operator<<(std::ostream & os, const ostream_beam_view & obv) {
    os << "p(" << obv.beam_view.p << ") eob(" << std::boolalpha << obv.beam_view.eob << ") tokens(";
    for (size_t i = 0 ; i < obv.beam_view.n_tokens ; ++i) {
        os << llama_token_to_piece(obv.ctx, obv.beam_view.tokens[i]);
    }
    return os << ')';
}

// Put here anything you want back in beam_search_callback().
struct beam_search_callback_data {
    llama_context * ctx;
    std::vector<llama_token> response;
};

// In this case, end-of-beam (eob) is equivalent to end-of-sentence (eos) but this need not always be the same.
// For example, eob can be flagged due to maximum token length, stop words, etc.
static bool is_at_eob(const beam_search_callback_data & callback_data, const llama_token * tokens, size_t n_tokens) {
    return n_tokens && llama_token_is_eog(llama_get_model(callback_data.ctx), tokens[n_tokens-1]);
}

// Function matching type llama_beam_search_callback_fn_t.
// Custom callback example is called each time the beams lengths increase:
//  * Show progress by printing ',' following by number of convergent beam tokens if any.
//  * When all beams converge to a common prefix, they are made available in beams_state.beams[0].
//    This is also called when the stop condition is met.
//    Collect tokens into std::vector<llama_token> response which is pointed to by callback_data.
static void beam_search_callback(void * callback_data_ptr, llama_beams_state beams_state) {
    auto& callback_data = *static_cast<beam_search_callback_data*>(callback_data_ptr);
    // Mark beams as EOS as needed.
    for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
        llama_beam_view& beam_view = beams_state.beam_views[i];
        if (!beam_view.eob && is_at_eob(callback_data, beam_view.tokens, beam_view.n_tokens)) {
            beam_view.eob = true;
        }
    }
    printf(",");  // Show progress
    if (const size_t n = beams_state.common_prefix_length) {
        callback_data.response.resize(callback_data.response.size() + n);
        assert(0u < beams_state.n_beams);
        const llama_token * tokens = beams_state.beam_views[0].tokens;
        std::copy(tokens, tokens + n, callback_data.response.end() - n);
        printf("%zu", n);
    }
    fflush(stdout);
#if 1 // DEBUG: print current beams for this iteration
    std::cout << "\n\nCurrent beams (last_call=" << beams_state.last_call << "):\n";
    for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
        std::cout << "beams["<<i<<"]: " << ostream_beam_view{callback_data.ctx,beams_state.beam_views[i]} << std::endl;
    }
#endif
}

int main(int argc, char ** argv)
{
    gpt_params params;
    //params.n_gpu_layers = 200;

    //---------------------------------
    // Print help :
    //---------------------------------

    if ( argc < 2 || argv[1][0] == '-' )
    {
        printf( "Usage: %s MODEL_PATH [BEAM_WIDTH=2] [PROMPT]\n" , argv[0] );
        return 1 ;
    }

    //---------------------------------
    // Load parameters :
    //---------------------------------

    params.model = argv[1];

    params.n_beams = 2 < argc ? std::stoi(argv[2]) : 2;

    if ( argc > 3 )
    {
        params.prompt = argv[3];
    }

    if ( params.prompt.empty() )
    {
        params.prompt = "### Request:\nHow many countries are there?\n\n### Response:\n";
    }

    //---------------------------------
    // Init LLM :
    //---------------------------------

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    std::tie(model, ctx) = llama_init_from_gpt_params( params );

    if ( model == NULL )
    {
        fprintf( stderr , "%s: error: unable to load model\n" , __func__ );
        return 1;
    }

    //---------------------------------
    // Tokenize the prompt :
    //---------------------------------

    std::vector<llama_token> tokens_list = llama_tokenize(ctx, params.prompt, true);

    const size_t max_context_size     = llama_n_ctx( ctx );
    const size_t max_tokens_list_size = max_context_size - 4 ;

    if (tokens_list.size() > max_tokens_list_size)
    {
        fprintf( stderr , "%s: error: prompt too long (%zu tokens, max %zu)\n" ,
             __func__ , tokens_list.size() , max_tokens_list_size );
        return 1;
    }

    fprintf( stderr, "\n\n" );

    // Print the tokens from the prompt :

    for( auto id : tokens_list )
    {
        std::cout << llama_token_to_piece(ctx, id);
    }
    std::cout << std::flush;

    int n_past = 0;

    if (llama_decode(ctx, llama_batch_get_one(tokens_list.data(), tokens_list.size(), n_past, 0)))
    {
        fprintf(stderr, "%s : failed to eval prompt.\n" , __func__ );
        return 1;
    }
    n_past += tokens_list.size();

    beam_search_callback_data callback_data{ctx, {}};
    size_t const beam_width = static_cast<size_t>(params.n_beams);
    int const n_predict = 256;
    llama_beam_search(ctx, beam_search_callback, &callback_data, beam_width, n_past, n_predict);

    std::cout << "\n\n";
    for (llama_token const token_id : callback_data.response) {
        std::cout << llama_token_to_piece(ctx,token_id);
    }
    std::cout << std::endl;

    llama_free( ctx );
    llama_free_model( model );

    llama_backend_free();

    return 0;
}
