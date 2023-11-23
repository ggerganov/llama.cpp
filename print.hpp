//template<typename T> void print_fields(const T& obj);

#include <iostream>
//#include <refl.hpp>
#include "llama.h"

REFL_TYPE(ggml_object)
REFL_END
REFL_TYPE(ggml_tensor)
REFL_END
REFL_TYPE(ggml_cplan )
REFL_END
REFL_TYPE(ggml_hash_set )
REFL_END
REFL_TYPE(ggml_cgraph )
REFL_END
REFL_TYPE(ggml_scratch )
REFL_END
REFL_TYPE(ggml_init_params )
REFL_END
REFL_TYPE(ggml_compute_params )
REFL_END
REFL_TYPE(ggml_opt_params )
REFL_END
REFL_TYPE(ggml_opt_params::ggml_adam)
REFL_END
REFL_TYPE(ggml_opt_params::ggml_lbfgs)
REFL_END
REFL_TYPE(ggml_opt_context )
REFL_END
REFL_TYPE(ggml_opt_context::ggml_grad )
REFL_END
REFL_TYPE(gpt_params )
REFL_END
REFL_TYPE(gguf_init_params )
REFL_END
REFL_TYPE(ggml_something )
REFL_END
REFL_TYPE(llama_sampling_context )
REFL_END
REFL_TYPE(llama_token_data )
REFL_END
REFL_TYPE(llama_model )
REFL_END
REFL_TYPE(llama_token_data_array )
REFL_END
REFL_TYPE(llama_batch )
REFL_END
REFL_TYPE(llama_model_params )
REFL_END
REFL_TYPE(llama_context_params )
REFL_END
REFL_TYPE(llama_context )
REFL_END
REFL_TYPE(llama_model_quantize_params )
REFL_END
REFL_TYPE(llama_grammar_element )
REFL_END
REFL_TYPE(llama_timings )
REFL_END
REFL_TYPE(llama_beam_view )
REFL_END
REFL_TYPE(llama_beams_state )
REFL_END

// // A simple struct with some fields and a function
// // A custom attribute to mark some fields as hidden
struct hidden : refl::attr::usage::field {};

// // Another struct with some fields and a function, using the custom attribute
// struct Person {
//     std::string name;
//     int age;
//     [[hidden]] std::string password;
//     void say_hello() const {
//         std::cout << "Hello, I'm " << name << " and I'm " << age << " years old.\n";
//     }
// };

// // A generic function to print out the fields of any object
template<typename T>
void print_fields(const T& ) {
//     // Get the type descriptor of the object
  constexpr auto type = refl::reflect<T>();
  
//     // Print the type name
//  std::cout << "DEBUG:" << TypeName<T>.fullname_intern() << "\n";
  std::cout << "DEBUG Type: " << type.name.c_str() << "\n";

  //  T instance{};
  for_each(refl::reflect<T>().members, [&](auto member) {

    std::cout <<     member.name.str() << "\n";
      
  });

     refl::util::for_each(type.members, [&](auto member) {
//         // Check if the member is a field and not hidden
       //if ((refl::descriptor::is_field(member)) && (!member.has_attribute<hidden>()))) {
       //if ((refl::descriptor::is_field(member))) {
//             // Print the member name and value
	 std::cout << member.name << ": " << "\n";
	 //	 refl::get(member, obj)
	 //}
     });
     std::cout << "\n";
}

