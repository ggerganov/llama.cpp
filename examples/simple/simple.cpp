#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"
#include "build-info.h"

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
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif



int main(int argc, char ** argv)
{
    gpt_params params;

    //---------------------------------
    // Print help :
    //---------------------------------

    if ( argc == 1 || argv[1][0] == '-' )
    {
        printf( "usage: %s MODEL_PATH [PROMPT]\n" , argv[0] );
        return 1 ;
    }

    //---------------------------------
    // Load parameters :
    //---------------------------------

    if ( argc >= 2 )
    {
        params.model = argv[1];
    }

    if ( argc >= 3 )
    {
        params.prompt = argv[2];
    }

    if ( params.prompt.empty() )
    {
        params.prompt = "Hello my name is";
    }

    //---------------------------------
    // Init LLM :
    //---------------------------------

    llama_backend_init(params.numa);

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

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize( ctx , params.prompt , true );

    const int max_context_size     = llama_n_ctx( ctx );
    const int max_tokens_list_size = max_context_size - 4 ;

    if ( (int)tokens_list.size() > max_tokens_list_size )
    {
        fprintf( stderr , "%s: error: prompt too long (%d tokens, max %d)\n" ,
             __func__ , (int)tokens_list.size() , max_tokens_list_size );
        return 1;
    }

    fprintf( stderr, "\n\n" );

    // Print the tokens from the prompt :

    for( auto id : tokens_list )
    {
        printf( "%s" , llama_token_to_str( ctx , id ) );
    }

    fflush(stdout);


    //---------------------------------
    // Main prediction loop :
    //---------------------------------

    // The LLM keeps a contextual cache memory of previous token evaluation.
    // Usually, once this cache is full, it is required to recompute a compressed context based on previous
    // tokens (see "infinite text generation via context swapping" in the main example), but in this minimalist
    // example, we will just stop the loop once this cache is full or once an end of stream is detected.

    while ( llama_get_kv_cache_token_count( ctx ) < max_context_size )
    {
        //---------------------------------
        // Evaluate the tokens :
        //---------------------------------

        if ( llama_eval( ctx , tokens_list.data() , int(tokens_list.size()) , llama_get_kv_cache_token_count( ctx ) , params.n_threads ) )
        {
            fprintf( stderr,  "%s : failed to eval\n" , __func__ );
            return 1;
        }

        tokens_list.clear();

        //---------------------------------
        // Select the best prediction :
        //---------------------------------

        llama_token new_token_id = 0;

        auto logits  = llama_get_logits( ctx );
        auto n_vocab = llama_n_vocab( ctx ); // the size of the LLM vocabulary (in tokens)

        std::vector<llama_token_data> candidates;
        candidates.reserve( n_vocab );

        for( llama_token token_id = 0 ; token_id < n_vocab ; token_id++ )
        {
            candidates.emplace_back( llama_token_data{ token_id , logits[ token_id ] , 0.0f } );
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // Select it using the "Greedy sampling" method :
        new_token_id = llama_sample_token_greedy( ctx , &candidates_p );


        // is it an end of stream ?
        if ( new_token_id == llama_token_eos() )
        {
            fprintf(stderr, " [end of text]\n");
            break;
        }

        // Print the new token :
        printf( "%s" , llama_token_to_str( ctx , new_token_id ) );
        fflush( stdout );

        // Push this new token for next evaluation :
        tokens_list.push_back( new_token_id );

    } // wend of main loop

    llama_free( ctx );
    llama_free_model( model );

    llama_backend_free();

    return 0;
}

// EOF
