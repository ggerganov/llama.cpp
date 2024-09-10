#include "arg.h"
#include "common.h"

#include <fstream>
#include <string>

// Export usage message (-h) to markdown format

static void export_md(std::string fname, llama_example ex) {
    std::ofstream file(fname, std::ofstream::out | std::ofstream::trunc);

    gpt_params params;
    auto ctx_arg = gpt_params_parser_init(params, ex);

    file << "| Argument | Explanation |\n";
    file << "| -------- | ----------- |\n";
    for (auto & opt : ctx_arg.options) {
        file << "| `";
        // args
        for (const auto & arg : opt.args) {
        if (arg == opt.args.front()) {
                file << arg;
                if (opt.args.size() > 1) file << ", ";
            } else {
                file << arg << (arg != opt.args.back() ? ", " : "");
            }
        }
        // value hint
        if (opt.value_hint) {
            std::string md_value_hint(opt.value_hint);
            string_replace_all(md_value_hint, "|", "\\|");
            file << " " << md_value_hint;
        }
        if (opt.value_hint_2) {
            std::string md_value_hint_2(opt.value_hint_2);
            string_replace_all(md_value_hint_2, "|", "\\|");
            file << " " << md_value_hint_2;
        }
        // help text
        std::string md_help(opt.help);
        string_replace_all(md_help, "\n", "<br/>");
        string_replace_all(md_help, "|", "\\|");
        file << "` | " << md_help << " |\n";
    }
}

int main(int, char **) {
    export_md("autogen-main.md", LLAMA_EXAMPLE_MAIN);
    export_md("autogen-server.md", LLAMA_EXAMPLE_SERVER);

    return 0;
}
