#include<string>

void process_output_plugin_ocaml_init();
void process_output_plugin_ocaml_destroy();
std::string process_output_plugin_ocaml(const std::string start,
					const std::string state,
					const std::string input) ;
