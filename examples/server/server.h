#include <httplib.h>
#include <json.hpp>
#include <cstring>
#include "common.h"
#include "llama.h"

/*
    This isn't the best way to do this.

    Missing:
        - Clean context (insert new prompt for change the behavior,
        this implies clean kv cache and emb_inp in runtime)
        - Release context (free memory) after shutdown the server

*/

class Llama{
    public:
        Llama(gpt_params params_) : params(params_){};
        bool load_context();
        bool prompt_test();
        void setting_context();
        int set_message(std::string msg);
        void release();

        llama_token nextToken();
        std::string inference();

        bool context_config = false;
        bool is_antiprompt = false;
        int tokens_completion = 0;
        gpt_params params;
        std::string user_tag = "### Human:", assistant_tag = "### Assistant:";
    private:
        llama_context *ctx;
        int n_ctx;
        int n_past = 0;
        int n_consumed = 0;
        int n_session_consumed = 0;
        int n_remain = 0;
        std::vector<llama_token> embd;
        std::vector<llama_token> last_n_tokens;
        bool is_interacting = false;
        std::vector<int> llama_token_newline;
        std::vector<int> embd_inp;

        // to ignore this in the completion
        std::vector<int> user_tag_tokens;
        std::vector<int> assistant_tag_tokens;
};
