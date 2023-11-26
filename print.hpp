#include <refl-cpp/refl.hpp>
#include <iostream>
#include "llama.h"
#include "common/common.h"
//#include "ggml-internal.hpp"
//#include "llama-internal.hpp"

REFL_TYPE(ggml_init_params )
REFL_END

// we use the named data type patch
#define ggml_opt_params_names
#ifdef ggml_opt_params_names
REFL_TYPE(ggml_opt_params::ggml_adam)
REFL_END

REFL_TYPE(ggml_opt_params::ggml_lbfgs)
REFL_END


REFL_TYPE(ggml_opt_context::ggml_grad )
REFL_END
#endif
REFL_TYPE(gpt_params )

REFL_FIELD( seed )
REFL_FIELD( n_threads)
REFL_FIELD( n_threads_batch)
REFL_FIELD( n_predict )
REFL_FIELD( n_ctx )
REFL_FIELD( n_batch)
REFL_FIELD( n_keep )
REFL_FIELD( n_draft)
REFL_FIELD( n_chunks )
REFL_FIELD( n_parallel)
REFL_FIELD( n_sequences)
REFL_FIELD( p_accept  )
REFL_FIELD( p_split )
REFL_FIELD( n_gpu_layers)
REFL_FIELD( n_gpu_layers_draft)
REFL_FIELD( main_gpu )
REFL_FIELD( tensor_split)
REFL_FIELD( n_beams )
REFL_FIELD(rope_freq_base)
REFL_FIELD( rope_freq_scale )
REFL_FIELD( yarn_ext_factor )
REFL_FIELD( yarn_attn_factor )
REFL_FIELD( yarn_beta_fast )
REFL_FIELD( yarn_beta_slow )
REFL_FIELD( yarn_orig_ctx)
REFL_FIELD( rope_scaling_type)
REFL_FIELD( sparams)
REFL_FIELD(model )
REFL_FIELD(model_draft )
REFL_FIELD(model_alias)
REFL_FIELD(prompt )
REFL_FIELD(prompt_file )
REFL_FIELD(path_prompt_cache )
REFL_FIELD(input_prefix )
REFL_FIELD(input_suffix )
REFL_FIELD( antiprompt)
REFL_FIELD(logdir )
REFL_FIELD( lora_adapter)
REFL_FIELD(lora_base )
REFL_FIELD( ppl_stride )
REFL_FIELD( ppl_output_type )
REFL_FIELD( hellaswag )
REFL_FIELD( hellaswag_tasks )
REFL_FIELD( mul_mat_q )
REFL_FIELD( memory_f16)
REFL_FIELD( random_prompt )
REFL_FIELD( use_color )
REFL_FIELD( interactive )
REFL_FIELD( chatml )
REFL_FIELD( prompt_cache_all )
REFL_FIELD( prompt_cache_ro )
REFL_FIELD( embedding )
REFL_FIELD( escape )
REFL_FIELD( interactive_first )
REFL_FIELD( multiline_input )
REFL_FIELD( simple_io )
REFL_FIELD( cont_batching )
REFL_FIELD( input_prefix_bos )
REFL_FIELD( ignore_eos )
REFL_FIELD( instruct )
REFL_FIELD( logits_all )
REFL_FIELD( use_mmap)
REFL_FIELD( use_mlock )
REFL_FIELD( numa )
REFL_FIELD( verbose_prompt )
REFL_FIELD( infill ) 
REFL_FIELD(mmproj )
REFL_FIELD( image)

REFL_END

REFL_TYPE(llama_sampling_params)
REFL_END

#ifdef llm_arch
REFL_TYPE(llm_arch)
REFL_END
#endif
REFL_TYPE(llama_sampling_context )
REFL_FIELD( params)
REFL_FIELD( mirostat_mu)
REFL_FIELD( grammar)
REFL_FIELD( parsed_grammar)
//REFL_FIELD( prev)  // TODO fixme has null data 
//REFL_FIELD( cur)
REFL_END

REFL_TYPE(llama_token_data )
REFL_END


REFL_TYPE(llama_token_data_array )
REFL_END

REFL_TYPE(llama_batch )
REFL_END


REFL_TYPE(ggml_object)
  REFL_FIELD(offs)
REFL_END

REFL_TYPE(ggml_tensor)
  REFL_FIELD(type)
  REFL_FIELD(type)
  REFL_FIELD(backend)
  REFL_FIELD(buffer)
  REFL_FIELD(n_dims)
  REFL_FIELD(ne)
  REFL_FIELD(nb)
  REFL_FIELD(op)
  REFL_FIELD(op_params)
  REFL_FIELD(is_param)
  REFL_FIELD(grad)
  REFL_FIELD(src)
  REFL_FIELD(perf_runs)
  REFL_FIELD(perf_cycles)
  REFL_FIELD(perf_time_us)
  REFL_FIELD(view_src)
  REFL_FIELD(view_offs)
  REFL_FIELD(data)
  REFL_FIELD(name)
  REFL_FIELD(extra)
  REFL_FIELD(padding)
REFL_END

REFL_TYPE(ggml_cplan)
  REFL_FIELD(work_size)
  REFL_FIELD(work_data)
  REFL_FIELD(n_threads)
  REFL_FIELD(abort_callback)
  REFL_FIELD(abort_callback_data)
REFL_END

REFL_TYPE(ggml_hash_set)
  REFL_FIELD(size)
REFL_END

REFL_TYPE(ggml_cgraph)
  REFL_FIELD(size)
  REFL_FIELD(n_nodes)
  REFL_FIELD(n_leafs)
  REFL_FIELD(nodes)
  REFL_FIELD(grads)
  REFL_FIELD(leafs)
  REFL_FIELD(visited_hash_table)
  REFL_FIELD(order)
  REFL_FIELD(perf_runs)
  REFL_FIELD(perf_cycles)
  REFL_FIELD(perf_time_us)
REFL_END

REFL_TYPE(ggml_scratch)
  REFL_FIELD(offs)
  REFL_FIELD(size)
  REFL_FIELD(data)

REFL_END

REFL_TYPE(ggml_compute_params)
  REFL_FIELD(type)
  REFL_FIELD(ith)
  REFL_FIELD(nth)
  REFL_FIELD(wsize)
  REFL_FIELD(wdata)

REFL_END

REFL_TYPE(ggml_opt_params)
  REFL_FIELD(type)
REFL_END

REFL_TYPE(ggml_opt_context)
  REFL_FIELD(ctx)
REFL_END

REFL_TYPE(gguf_init_params)
REFL_END

REFL_TYPE(ggml_something)
  REFL_FIELD(type_name)
REFL_END

#ifdef ggml_context
REFL_TYPE(ggml_context)
  REFL_FIELD(mem_size)
REFL_FIELD(mem_buffer)
REFL_FIELD(mem_buffer_owned)
REFL_FIELD(    no_alloc)
REFL_FIELD(    no_alloc_save)
REFL_FIELD(    n_objects)
REFL_FIELD(    objects_begin)
REFL_FIELD(    objects_end)
REFL_FIELD(    scratch)
REFL_FIELD(    scratch_save)
REFL_END
#endif

#ifdef ggml_context_container
REFL_TYPE(ggml_context_container)
  REFL_FIELD(used)
  REFL_FIELD(context)
REFL_END
#endif

#ifdef ggml_numa_node
 REFL_TYPE(ggml_numa_node)
   REFL_FIELD(cpus)
   REFL_FIELD(n_cpus)
 REFL_END

 REFL_TYPE(ggml_numa_nodes)
   REFL_FIELD(nodes)
   REFL_FIELD(n_nodes)
 REFL_END

 REFL_TYPE(ggml_state)
   REFL_FIELD(contexts)
   REFL_FIELD(numa)
   REFL_END

 REFL_TYPE(gguf_str)
   REFL_FIELD(n)
   REFL_FIELD(data)
 REFL_END

 REFL_TYPE(ggml_map_custom1_op_params)
   REFL_FIELD(fun)
   REFL_FIELD(n_tasks)
 REFL_END

REFL_TYPE(ggml_map_custom2_op_params)
  REFL_FIELD(fun)
  REFL_FIELD(n_tasks)
REFL_END

REFL_TYPE(ggml_map_custom3_op_params)
  REFL_FIELD(fun)
  REFL_FIELD(n_tasks)
REFL_END

REFL_TYPE(hash_map)
  REFL_FIELD(set)
  REFL_FIELD(vals)
REFL_END

REFL_TYPE(ggml_compute_state_shared)
  REFL_FIELD(cgraph)
  REFL_FIELD(cplan)
REFL_END
REFL_TYPE(ggml_compute_state)
  REFL_FIELD(thrd)
  REFL_FIELD(ith)
REFL_END
REFL_TYPE(ggml_lbfgs_iteration_data)
  REFL_FIELD(alpha)
  REFL_FIELD(ys)
REFL_END
#endif

#ifdef gguf_kv
REFL_TYPE(gguf_kv)
  REFL_FIELD(key)
  REFL_FIELD(type)
REFL_END


REFL_TYPE(gguf_header)
  REFL_FIELD(magic)
  REFL_FIELD(version)
REFL_END

REFL_TYPE(gguf_tensor_info)
  REFL_FIELD(name)
  REFL_FIELD(n_dims)
REFL_END

REFL_TYPE(gguf_context)
  REFL_FIELD(header)
  REFL_FIELD(kv)
REFL_END

REFL_TYPE(gguf_buf)
  REFL_FIELD(data)
  REFL_FIELD(size)
REFL_END
#endif

REFL_TYPE(llama_model_params)
  REFL_FIELD(n_gpu_layers)
REFL_END
REFL_TYPE(llama_context_params)
  REFL_FIELD(seed)
REFL_END

REFL_TYPE(llama_model_quantize_params)
  REFL_FIELD(nthread)
REFL_END

REFL_TYPE(llama_grammar_element)
REFL_END

REFL_TYPE(llama_timings)
  REFL_FIELD(t_start_ms)
REFL_END
REFL_TYPE(llama_beam_view)
  REFL_FIELD(tokens)
REFL_END

REFL_TYPE(llama_beams_state)
  REFL_FIELD(beam_views)
REFL_END

#ifdef ggml_backend
REFL_TYPE(ggml_backend)
REFL_END
#endif

REFL_TYPE(ggml_backend_buffer)
REFL_END

#ifdef ggml_allocr
REFL_TYPE(ggml_allocr)
REFL_END

REFL_TYPE(ggml_tallocr)
REFL_END

REFL_TYPE(ggml_gallocr)
REFL_END

#endif

#ifdef llama_buffer
REFL_TYPE(llama_buffer)
REFL_FIELD(data)
REFL_FIELD(size)
REFL_END
  
REFL_TYPE(llama_file)
REFL_FIELD(fp)
REFL_FIELD(size)
REFL_END
  

REFL_TYPE(llama_mmap)
REFL_FIELD(addr)
REFL_FIELD(size)
REFL_END


REFL_TYPE(llama_mlock)
  REFL_FIELD(addr)
  REFL_FIELD(size)
REFL_END

REFL_TYPE(llama_state)
 REFL_FIELD(log_callback)
 REFL_FIELD(log_callback_user_data)
 REFL_END
  

REFL_TYPE(llama_hparams)
  REFL_FIELD(vocab_only)
  REFL_FIELD(n_vocab)
  REFL_END


REFL_TYPE(llama_cparams)
  REFL_FIELD(n_ctx)
  REFL_FIELD(n_batch)
REFL_END

REFL_TYPE(llama_layer)
 REFL_FIELD(attn_norm)
 REFL_FIELD(attn_norm_b)
REFL_END

REFL_TYPE(llama_kv_cell)
  REFL_FIELD(pos)
  REFL_FIELD(delta)
REFL_END

REFL_TYPE(llama_kv_cache)
   REFL_FIELD(has_shift)
   REFL_FIELD(head)
 REFL_END
#endif

#ifdef e_model
REFL_TYPE(e_model)
REFL_END
#endif

REFL_TYPE(llama_ftype)
REFL_END

//#ifdef llama_model
REFL_TYPE(llama_model)
REFL_FIELD(type)
REFL_FIELD(arch)
REFL_FIELD(ftype )
REFL_FIELD(name )
REFL_FIELD(hparams )
REFL_FIELD(vocab)
REFL_FIELD(tok_embd)
REFL_FIELD(pos_embd)
REFL_FIELD(tok_norm)
REFL_FIELD(tok_norm_b)
REFL_FIELD(output_norm)
REFL_FIELD(output_norm_b)
REFL_FIELD(output)
REFL_FIELD(layers)
REFL_FIELD(n_gpu_layers)
REFL_FIELD(gguf_kv) //unordered map
REFL_FIELD(ctx)
REFL_FIELD(buf)
REFL_FIELD(mapping) //std::unique_ptr 
REFL_FIELD(mlock_buf)
REFL_FIELD(mlock_mmap)
REFL_FIELD(tensors_by_name)
REFL_FIELD(t_load_us)
REFL_FIELD(t_start_us)
REFL_END
//#endif

#ifdef llama_vocab
REFL_TYPE(llama_vocab)
  REFL_END
#endif

REFL_TYPE(grammar_parser::parse_state)
REFL_END

//#ifdef llama_context
REFL_TYPE(llama_context)
REFL_FIELD( cparams)
//REFL_FIELD(model)
REFL_FIELD(kv_self)
REFL_FIELD(rng) //random numbers
REFL_FIELD(has_evaluated_once )
REFL_FIELD( t_start_us)
REFL_FIELD( t_load_us)
REFL_FIELD( t_sample_us )
REFL_FIELD( t_p_eval_us )
REFL_FIELD( t_eval_us)
REFL_FIELD( n_sample )
REFL_FIELD( n_p_eval )
REFL_FIELD( n_eval  )
//REFL_FIELD(  logits) crash
REFL_FIELD(  logits_all )
REFL_FIELD(  embedding)
//REFL_FIELD(   work_buffer)
REFL_FIELD(   buf_compute)
REFL_FIELD( buf_alloc)
REFL_FIELD( alloc ) 
#ifdef GGML_USE_METAL
REFL_FIELD( ctx_metal )
#endif
#ifdef GGML_USE_MPI
REFL_FIELD( ctx_mpi )
#endif
REFL_END
//#endif

#ifdef llama_model_loader
REFL_TYPE(llama_model_loader)
  REFL_FIELD(n_kv)
  REFL_FIELD(n_tensors)
REFL_END
#endif

#ifdef llm_build_context
REFL_TYPE(llm_build_context)
// REFL_FIELD(model) cannot create pointer to reference member ‘llm_build_context::model’
//  REFL_FIELD(hparams) cannot create pointer to reference member ‘llm_build_context::hparams’
REFL_END

REFL_TYPE(llm_offload_trie)
REFL_END

REFL_TYPE(llm_symbol)
  REFL_FIELD(prev)
REFL_END

REFL_TYPE(llm_bigram_spm)
REFL_END

REFL_TYPE(llm_tokenizer_spm)
REFL_END

REFL_TYPE(llm_bigram_bpe)
REFL_END

REFL_TYPE(llm_tokenizer_bpe)
REFL_END
  

REFL_TYPE(fragment_buffer_variant)
REFL_END
  

REFL_TYPE(llama_partial_utf8)
  REFL_FIELD(value)
  REFL_FIELD(n_remain)
REFL_END
  

REFL_TYPE(llama_grammar)
 REFL_FIELD(rules)
 REFL_FIELD(stacks)
REFL_END
  

REFL_TYPE(llama_grammar_candidate)
 REFL_FIELD(index)
 REFL_FIELD(code_points)
REFL_END
  

REFL_TYPE(llama_beam)
  REFL_FIELD(tokens)
  REFL_FIELD(p)
REFL_END
  

REFL_TYPE(llama_logit_info)
//  REFL_FIELD(logits)
  REFL_FIELD(n_vocab)
REFL_END

REFL_TYPE(llama_beam_search_data)
  REFL_FIELD(ctx)
  REFL_FIELD(n_beams)
REFL_END


REFL_TYPE(quantize_state_internal)
//  REFL_FIELD(model)
  REFL_FIELD(params)
REFL_FIELD( n_attention_wv )
REFL_FIELD(    n_feed_forward_w2 )
  REFL_FIELD(    i_attention_wv    )
  REFL_FIELD(    i_feed_forward_w2 )
REFL_FIELD(    n_k_quantized     )
REFL_FIELD(     n_fallback        )
REFL_END

REFL_TYPE(llama_data_context)
REFL_END
  
REFL_TYPE(llama_data_buffer_context)
  REFL_FIELD(ptr)
REFL_END

REFL_TYPE(llama_data_file_context)
  REFL_FIELD(file)
REFL_END
#endif

template <typename T>
constexpr auto get_value_type_name(const T t) noexcept
{
  return t.value_type;
}

namespace runtime2
    {
      using namespace refl;
      using namespace refl::descriptor;
      template <typename CharT, typename T>
        void debug(std::basic_ostream<CharT>& os, const T& value, bool compact = false);

        namespace detail
        {
            template <typename CharT, typename T, typename = decltype(std::declval<std::basic_ostream<CharT>&>() << std::declval<T>())>
            std::true_type is_ostream_printable_test(int);

            template <typename CharT, typename T>
            std::false_type is_ostream_printable_test(...);

            template <typename CharT, typename T>
            constexpr bool is_ostream_printable_v{ decltype(is_ostream_printable_test<CharT, T>(0))::value };

            namespace
            {
                [[maybe_unused]] int next_depth(int depth)
                {
                    return depth == -1 || depth > 8
                        ? -1
                        : depth + 1;
                }
            }

            template <typename CharT>
            void indent(std::basic_ostream<CharT>& os, int depth)
            {
                for (int i = 0; i < depth; i++) {
                    os << "    ";
                }
            }

            template <typename CharT, typename T>
            void debug_impl(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] int depth);

            template <typename CharT, typename T>
            void debug_detailed(std::basic_ostream<CharT>& os, const T& value, int depth)
            {

                using type_descriptor = type_descriptor<T>;
                bool compact = depth == -1;
                // print type with members enclosed in braces
                os << type_descriptor::name << " { ";
                if (!compact) os << '\n';

                constexpr auto readable_members = filter(type_descriptor::members, [](auto member) { return is_readable(member); });
                for_each(readable_members, [&](auto member, [[maybe_unused]] auto index) {
                    int new_depth = next_depth(depth);

                    indent(os, new_depth);
                    os << get_display_name(member) << " = ";

                    if constexpr (util::contains_instance<attr::debug>(member.attributes)) {
                        // use the debug attribute to print
                        auto debug_attr = util::get_instance<attr::debug>(member.attributes);
                        debug_attr.write(os, value);
                    }
                    else {
                        debug_impl(os, member(value), new_depth);
                    }

                    if (!compact || index + 1 != readable_members.size) {
                        os << ", ";
                    }
                    if (!compact) {
                        indent(os, depth);
                        os << '\n';
                    }
                });

                if (compact) os << ' ';
                indent(os, depth);
                os << '}';
            }

            template <typename CharT, typename T>
            void debug_reflectable(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] int depth)
            {
                using type_descriptor = type_descriptor<T>;
                if constexpr (trait::contains_instance_v<attr::debug, typename type_descriptor::attribute_types>) {
                    // use the debug attribute to print
                    auto debug_attr = util::get_instance<attr::debug>(type_descriptor::attributes);
                    debug_attr.write(os, value);
                }
                else if constexpr (detail::is_ostream_printable_v<CharT, T>) {
                    // type supports printing natively, just use that

		  os << value;

                }
                else {
                    debug_detailed(os, value, depth);
                }
            }

            template <typename CharT, typename T>
            void debug_container(std::basic_ostream<CharT>& os, const T& value, int depth)
            {
                bool compact = depth == -1;
                os << "[";

                auto end = value.end();
                for (auto it = value.begin(); it != end; ++it)
                {
                    if (!compact) os << '\n';
                    int new_depth = next_depth(depth);
                    indent(os, new_depth);
		    
		      debug_impl(os, *it, new_depth);
		      if (std::next(it, 1) != end) {
                        os << ", ";
		      }
		      else if (!compact) {
                        os << '\n';
		      }
		    
                }

                indent(os, depth);
                os << "]";
            }

            template <typename CharT, typename T>
            void debug_impl(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] int depth)
            {
                using no_pointer_t = std::remove_pointer_t<T>;

                if constexpr (std::is_same_v<bool, T>) {
                    os << (value ? "true" : "false");
                }
                else if constexpr (std::is_pointer_v<T> && !std::is_void_v<no_pointer_t> && trait::is_reflectable_v<no_pointer_t>) {
                    if (value == nullptr) {
                        os << "nullptr";
                    }
                    else {
                        os << '&';
                        debug_impl(os, *value, -1);
                    }
                }
                else if constexpr (trait::is_reflectable_v<T>) {
                    debug_reflectable(os, value, depth);
                }
                else if constexpr (detail::is_ostream_printable_v<CharT, T>) {
                    os << value;
                }
                else if constexpr (trait::is_container_v<T>) {
                    debug_container(os, value, depth);
                }
                else {
                    os << "(not printable)";
                }
            }
        }

        /**
         * Writes the debug representation of value to the given std::ostream.
         * Calls the function specified by the debug<F> attribute whenever possible,
         * before falling back to recursively interating the members and printing them.
         * Takes an optional arguments specifying whether to print a compact representation.
         * The compact representation contains no newlines.
         */
        template <typename CharT, typename T>
        void debug(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] bool compact)
        {
            static_assert(trait::is_reflectable_v<T> || trait::is_container_v<T> || detail::is_ostream_printable_v<CharT, T>,
                "Type is not reflectable, not a container of reflectable types and does not support operator<<(std::ostream&, T)!");

            detail::debug_impl(os, value, compact ? -1 : 0);
        }

        /**
         * Writes the compact debug representation of the provided values to the given std::ostream.
         */
        template <typename CharT, typename... Ts>
        void debug_all(std::basic_ostream<CharT>& os, const Ts&... values)
        {
            refl::runtime::debug(os, std::forward_as_tuple(static_cast<const Ts&>(values)...), true);
        }

        /**
         * Writes the debug representation of the provided value to an std::string and returns it.
         * Takes an optional arguments specifying whether to print a compact representation.
         * The compact representation contains no newlines.
         */
        template <typename CharT = char, typename T>
        std::basic_string<CharT> debug_str(const T& value, bool compact = false)
        {
            std::basic_stringstream<CharT> ss;
            debug(ss, value, compact);
            return ss.str();
        }

        /**
         * Writes the compact debug representation of the provided values to an std::string and returns it.
         */
        template <typename CharT = char, typename... Ts>
        std::basic_string<CharT> debug_all_str(const Ts&... values)
        {
            return refl::runtime::debug_str(std::forward_as_tuple(static_cast<const Ts&>(values)...), true);
        }
}

// // A generic function to print out the fields of any object
template<typename T>
void print_fields(const T& t) {
  runtime2::debug(std::cout, t);
  constexpr auto type = refl::reflect<T>();

  // constexpr auto membertype = refl::member_list<T>();

  // constexpr auto members = get_members(type);
  // std::cout << "DEBUG Type: " << type.name.c_str() << "\n";
  // std::cout << "DEBUG Type2: " << typeid(membertype).name() << "\n";
  // std::cout << "DEBUG Type3: " << typeid(members).name() << "\n";
  //    refl::util::for_each(members, [&](auto member) {
  //      //using member_t = decltype(member::value_type);
  //      //typename type3 = member::value_type;
  //      //typename trait::remove_qualifiers_t<member_t>::value_type>;
  //      //constexpr auto type2 = refl::reflect(type3);
  // 	 //std::cout  << "Auto:" << foo <<"\n";       
  //      //std::cout  << "Auto:" << member.name <<"\n";
  //      //std::cout << "DEBUG Type2: " << typeid(member_t).name() << "\n";
  //      //std::cout << "DEBUG Type2: " << type2.name.c_str() << "\n";
  //    });
  //    //std::cout << "\n";
}
