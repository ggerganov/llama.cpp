#ifdef NDEBUG
#undef NDEBUG
#endif


#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unistd.h>

static std::string read(const std::string & file) {
    std::ostringstream actuals;
    actuals << std::ifstream(file.c_str()).rdbuf();
    return actuals.str();
}

static void assert_equals(const std::string & expected, const std::string & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

static void assert_contains(const std::string & expected, const std::string & actual) {
    if (actual.find(expected) == std::string::npos) {
        std::cerr << "Expected to find: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

struct Out {
    std::string out;
    std::string err;
};

static Out run(const std::string & cmd) {
    auto full_cmd = cmd + " > out/out.txt 2> out/err.txt";
    std::cerr << "Running: " << full_cmd << std::endl;
    auto out = read("out/out.txt");
    auto err = read("out/err.txt");
    if (std::system(full_cmd.c_str()) != 0)
        throw std::runtime_error("llama-cli binary failed to run.\nstdout: " + out + "\nstderr: " + err);
    return {
        /* .out = */ out,
        /* .err = */ err,
    };
}

int main(int argc, char ** argv) {
    std::string cli_bin = argc == 2 ? argv[1] : "./llama-cli";

    try {
        if (std::system("mkdir -p out/") != 0)
            throw std::runtime_error("Failed to create out/ directory.");

        {
            auto p = run(cli_bin + " --help");
            if (!p.err.empty())
                throw std::runtime_error("llama-cli --help should not have any stderr.");
            assert_contains("example usage", p.out);
        }

        {
            auto p = run(cli_bin + " -hfr ggml-org/models -hff tinyllamas/stories260K.gguf --prompt hello --seed 42 --samplers top-k --top-k 1 -ngl 0 -n 10");
            assert_equals(" hello was a big, red ball. He", p.out);
            assert_contains("system_info:", p.err);
        }

        {
            auto p = run(cli_bin + " -hfr ggml-org/models -hff tinyllamas/stories260K.gguf --prompt hello --seed 42 --samplers top-k --top-k 1 -ngl 0 -n 10 --log-disable");
            assert_equals(" hello was a big, red ball. He", p.out);
            assert_equals("", p.err);
        }

        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "[test-cli] Error: " << ex.what() << std::endl;
        return 1;
    }
}
