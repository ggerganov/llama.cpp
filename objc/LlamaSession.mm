#import <Foundation/Foundation.h>
#import "LlamaSession_Private.hpp"
#import "../../common/common.h"
#import "LlamaModel_Private.hpp"
#import "LlamaContext_Private.hpp"
#import "CPUParams_Private.hpp"
#import "GPTSampler.h"
#import <OSLog/OSLog.h>
#import "ggml.h"
#import "GPTParams_Private.hpp"
#import "GGMLThreadpool_Private.hpp"
#import "LlamaBatch_Private.hpp"

@implementation BlockingLineQueue {
    // Input queue and related synchronization
    NSMutableArray<NSString *> *inputQueue;
    NSCondition *inputCondition;

    // Output queue and related synchronization
    NSMutableArray<NSString *> *outputQueue;
    NSCondition *outputCondition;
    
    // Log queue
    NSMutableArray<NSString *> *log;
}

- (instancetype)init {
    if (self = [super init]) {
        inputQueue = [NSMutableArray new];
        outputQueue = [NSMutableArray new];
        log = [NSMutableArray new];
        inputCondition = [[NSCondition alloc] init];
        outputCondition = [[NSCondition alloc] init];
    }
    return self;
}

- (void)addInputLine:(NSString *)line {
    [inputCondition lock];
    [inputQueue addObject:line];
    [log addObject:line];
    [inputCondition signal]; // Notify that a new input line is available
    [inputCondition unlock];
}

- (NSString *)inputLine {
    [inputCondition lock];
    while ([inputQueue count] == 0) {
        [inputCondition wait];
    }
    NSString *line = [inputQueue objectAtIndex:0];
    [inputQueue removeObjectAtIndex:0];
    [inputCondition unlock];
    return line;
}

- (void)addOutputLine:(NSString *)line {
    [outputCondition lock];
    [outputQueue addObject:line];
    [log addObject:line];
    [outputCondition signal]; // Notify that a new output line is available
    [outputCondition unlock];
}

- (NSString *)outputLine {
    [outputCondition lock];
    while ([outputQueue count] == 0) {
        [outputCondition wait];
    }
    NSString *line = [outputQueue objectAtIndex:0];
    [outputQueue removeObjectAtIndex:0];
    [outputCondition unlock];
    return line;
}

- (void)dealloc
{
    [outputCondition unlock];
    [inputCondition unlock];
    [inputCondition dealloc];
    [outputCondition dealloc];
    [log dealloc];
    [inputQueue dealloc];
    [outputQueue dealloc];
    [super dealloc];
}
@end

@implementation LlamaSession {
    std::vector<llama_token> embd_inp;
    std::vector<common_chat_msg> chat_msgs;

    BOOL isInteracting;
    
    bool is_antiprompt;
    bool input_echo;
    bool display;
    bool need_to_save_session;
    
    int n_past;
    int n_remain;
    int n_consumed;
    int n_session_consumed;
    
    std::vector<int>   input_tokens;
    std::vector<int>   output_tokens;;
    std::ostringstream output_ss;
    std::stringstream last_output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode
    
    std::vector<llama_token> embd;
    NSMutableString *pathSession;
    NSInteger ga_i;
    NSInteger ga_n;
    NSInteger ga_w;
    std::vector<llama_token> session_tokens;
    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;
    BOOL need_insert_eot;
    int n_ctx;
    os_log_t os_log_inst;
    
    GGMLThreadpool *threadpool;
    GGMLThreadpool *threadpool_batch;
}

- (NSString *)chat_add_and_format:(std::vector<common_chat_msg> &) chat_msgs role:(const std::string &) role content:(const std::string &) content {
    common_chat_msg new_msg{role, content};
    auto formatted = common_chat_format_single([self.model cModel], [_params params].chat_template, chat_msgs, new_msg, role == "user");
    chat_msgs.push_back({role, content});
    os_log_debug(os_log_inst, "formatted: '%s'\n", formatted.c_str());
    return [NSString stringWithCString:formatted.c_str() encoding:NSUTF8StringEncoding];
}

static BOOL file_is_empty(NSString *path) {
    NSFileManager *manager = [NSFileManager defaultManager];
    if ([manager fileExistsAtPath:path]) {
        NSDictionary *attributes = [manager attributesOfItemAtPath:path error:nil];
        unsigned long long size = [attributes fileSize];
        if (attributes && size == 0) {
            return true;
        } else {
            return false;
        }
    }
    return true;
}

- (instancetype)initWithParams:(GPTParams *)params {
    self = [super init];
    self->_params = [params copy];
    self->_mutableLastOutput = [[NSMutableString alloc] init];
    self->_queue = [BlockingLineQueue new];
    if (params.logging) {
        os_log_inst = OS_LOG_DEFAULT;
    } else {
        os_log_inst = OS_LOG_DISABLED;
    }
    if (!params.modelPath) {
        [NSException raise:@"ModelFailure"
                    format:@"params.modelPath must be defined"];
    }

    os_log_info(os_log_inst,
                "%s: llama threadpool init, n_threads = %ld\n",
                __func__, static_cast<long>(params.cpuParams.nThreads));

    if (params.embedding) {
        os_log_error(os_log_inst,
                     R"(************
                     please use the 'embedding' tool for embedding calculations
                     ************)");
        abort();
    }

    if (params.nCtx != 0 && params.nCtx < 8) {
        os_log_info(os_log_inst, "minimum context size is 8, using minimum size.");
        params.nCtx = 8;
    }

    if (params.ropeFreqBase != 0) {
        os_log_info(os_log_inst, "changing RoPE frequency base to \(params.ropeFreqBase)");
    }

    if (params.ropeFreqScale != 0.0) {
        os_log_info(os_log_inst, "scaling RoPE frequency by \(params.ropeFreqScale)");
    }

    llama_backend_init();
    llama_numa_init(ggml_numa_strategy(params.numaStrategy));
    auto llama_init = common_init_from_params([params params]);
    if (llama_init.context == nil) {
        [NSException raise:@"ContextFailure"
                    format:@"could not load context"];
    }
    if (llama_init.model == nil) {
        [NSException raise:@"ModelLoadFailure"
                    format:@"could not load model"];
    }
    auto tpp_batch = params.cpuParamsBatch.ggmlThreadpoolParams;
    auto tpp = params.cpuParams.ggmlThreadpoolParams;

    set_process_priority(ggml_sched_priority(params.cpuParams.priority));
    
    if (tpp != tpp_batch) {
        threadpool_batch = [tpp_batch threadpool];
        if (!threadpool_batch) {
            [NSException raise:@"ThreadpoolFailure"
                        format:@"batch threadpool create failed"];
        }
        
        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }
    
    threadpool = [tpp threadpool];
    if (!threadpool) {
        [NSException raise:@"ThreadpoolFailure"
                    format:@"threadpool create failed"];
    }
    
    self.ctx = [[LlamaContext alloc] initWithContext:llama_init.context];
    [self.ctx attachThreadpool:threadpool threadpoolBatch:threadpool_batch];
    self.model = [[LlamaModel alloc] init:llama_init.model];
    const int n_ctx_train = [self.model nCtxTrain];
    n_ctx = [self.ctx nCtx];
    //
    if (n_ctx > n_ctx_train) {
        os_log_info(os_log_inst, "%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // print chat template example in conversation mode
    if (params.conversation) {
        if (params.enableChatTemplate) {
            os_log_info(os_log_inst, "%s: chat template example:\n%s\n", __func__,
                        [[self.model formatExample:params.chatTemplate] cStringUsingEncoding:NSUTF8StringEncoding]);
        } else {
            os_log_info(os_log_inst, "%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }
    // print system information
    @autoreleasepool {
        NSLog(@"%s", common_params_get_system_info([params params]).c_str());
    }
    
    pathSession = [[NSMutableString alloc] initWithString:params.pathPromptCache];
    
    NSFileManager *fileManager = [NSFileManager defaultManager];
    
    if ([pathSession length] != 0) {
        os_log_info(os_log_inst, "%s: attempting to load saved session from '%s'\n", __func__, [pathSession cStringUsingEncoding:NSUTF8StringEncoding]);
        if (![fileManager fileExistsAtPath:pathSession]) {
            os_log_info(os_log_inst, "%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(pathSession)) {
            os_log_info(os_log_inst,"%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (![self.ctx loadStateFile:pathSession tokensOut:session_tokens.data() nTokenCpacity:session_tokens.capacity() nTokenCountOut:&n_token_count_out]) {
                [NSException raise:@"SessionLoadFailure" format:@"%s: failed to load session file '%s'\n", __func__, [pathSession cStringUsingEncoding:NSUTF8StringEncoding]];
            }
            session_tokens.resize(n_token_count_out);
            os_log_info(os_log_inst,"%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }
    
    BOOL addBOS = [self.model addBOSToken];
    if (![self.model hasEncoder]) {
        GGML_ASSERT(![self.model addEOSToken]);
    }
    
    os_log_debug(os_log_inst, "n_ctx: %d, add_bos: %d\n", n_ctx, addBOS);
    
    
    {
        auto prompt = (params.conversation && params.enableChatTemplate && params.prompt.length > 0)
        ? [self chat_add_and_format:chat_msgs role:"system" content:[params params].prompt] // format the system prompt in conversation mode
        : params.prompt;
        if (params.interactiveFirst || [params.prompt length] > 0 || session_tokens.empty()) {
            os_log_debug(os_log_inst, "tokenize the prompt\n");
            embd_inp = [self.ctx tokenize:prompt addSpecial:true parseSpecial:true];
        } else {
            os_log_debug(os_log_inst,"use session tokens\n");
            embd_inp = session_tokens;
        }
        
        os_log_debug(os_log_inst,"prompt: \"%s\"\n", [prompt cStringUsingEncoding:NSUTF8StringEncoding]);
        os_log_debug(os_log_inst,"tokens: %s\n", [self.ctx convertTokensToString:embd_inp].c_str());
    }

    // Should not run without any tokens
    if (embd_inp.empty()) {
        if (addBOS) {
            embd_inp.push_back([self.model tokenBOS]);
            os_log_info(os_log_inst, "embd_inp was considered empty and bos was added: %s\n", [_ctx convertTokensToString:embd_inp].c_str());
        } else {
            [NSException raise:@"InputEmptyError" format:@"input is empty"];
        }
    }
    
    // Tokenize negative prompt
    if (embd_inp.size() > n_ctx - 4) {
        [NSException raise:@"PromptError" format:@"%s: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4];
    }
    
    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if ([params.prompt length] == 0 && n_matching_session_tokens == embd_inp.size()) {
            os_log_info(os_log_inst, "%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            os_log_info(os_log_inst, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            os_log_error(os_log_inst, "%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            os_log_info(os_log_inst, "%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }
        
        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm([self.ctx cContext], -1, n_matching_session_tokens, -1);
    }

    os_log_debug(os_log_inst, "recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
             embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        os_log_debug(os_log_inst, "recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);
        
        session_tokens.resize(embd_inp.size() - 1);
    }
    
    // number of tokens to keep when resetting context
    if (params.nKeep < 0 || params.nKeep > (int) embd_inp.size()) {
        params.nKeep = (int)embd_inp.size();
    } else {
        params.nKeep += addBOS; // always keep the BOS token
    }
    
    if (params.conversation) {
        params.interactiveFirst = true;
    }
    
    // enable interactive mode if interactive start is specified
    if (params.interactiveFirst) {
        params.interactive = true;
    }
    
    if (params.verbosePrompt) {
        os_log_info(os_log_inst,
                    "%s: prompt: '%s'\n", __func__, [params.prompt cStringUsingEncoding:NSUTF8StringEncoding]);
        os_log_info(os_log_inst, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            os_log_info(os_log_inst, "%6d -> '%s'\n", embd_inp[i],
                        [[self.ctx tokenToPiece:embd_inp[i]] cStringUsingEncoding:NSUTF8StringEncoding]);
        }
        
        if (params.nKeep > addBOS) {
            os_log_info(os_log_inst, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.nKeep; i++) {
                os_log_debug(os_log_inst, "%s",
                             [[self.ctx tokenToPiece:embd_inp[i]] cStringUsingEncoding:NSUTF8StringEncoding]);
            }
        }
    }

    if (params.interactive) {
        os_log_info(os_log_inst, "%s: interactive mode on.\n", __func__);
        
        if ([params.antiPrompts count] > 0) {
            for (NSString *antiprompt in params.antiPrompts) {
                os_log_info(os_log_inst, "Reverse prompt: '%s'\n", [antiprompt cStringUsingEncoding:NSUTF8StringEncoding]);
                if (params.verbosePrompt) {
                    auto tmp = [_ctx tokenize:antiprompt
                                  addSpecial:false
                                parseSpecial:true];
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        os_log_info(os_log_inst, "%6d -> '%s'\n", tmp[i], [[self.ctx tokenToPiece:tmp[i]] cStringUsingEncoding:NSUTF8StringEncoding]);
                    }
                }
            }
        }
        
        if (params.inputPrefixBOS) {
            os_log_info(os_log_inst, "Input prefix with BOS\n");
        }
        
        if ([params.inputPrefix length] > 0) {
            os_log_info(os_log_inst, "Input prefix: '%s'\n", [params.inputPrefix cStringUsingEncoding:NSUTF8StringEncoding]);
            if (params.verbosePrompt) {
                auto tmp = [_ctx tokenize:params.inputPrefix addSpecial:true parseSpecial:true];
                for (int i = 0; i < (int) tmp.size(); i++) {
                    os_log_info(os_log_inst, "%6d -> '%s'\n",
                                tmp[i], [[self.ctx tokenToPiece:tmp[i]] cStringUsingEncoding:NSUTF8StringEncoding]);
                }
            }
        }
        
        if ([params.inputSuffix length] > 0) {
            os_log_info(os_log_inst, "Input suffix: '%s'\n", [params.inputSuffix cStringUsingEncoding:NSUTF8StringEncoding]);
            if (params.verbosePrompt) {
                auto tmp = [_ctx tokenize:params.inputSuffix addSpecial:false parseSpecial:true];
                for (int i = 0; i < (int) tmp.size(); i++) {
                    os_log_info(os_log_inst, "%6d -> '%s'\n",
                                tmp[i], [[self.ctx tokenToPiece:tmp[i]] cStringUsingEncoding:NSUTF8StringEncoding]);
                }
            }
        }
    }
    
    _smpl = [[GPTSampler alloc] init:_model gptSamplerParams:[params samplerParams]];
    if (!_smpl) {
        [NSException raise:@"SamplingFailure" format:@"failed to initialize sampling subsystem"];
    }
    
    os_log_info(os_log_inst, "sampler seed: %u\n", [_smpl seed]);
    os_log_info(os_log_inst, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.nBatch, params.nPredict, params.nKeep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    
    ga_n = params.grpAttnN;
    ga_w = params.grpAttnW;
    
    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
        GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        os_log_info(os_log_inst, "self-extend: n_ctx_train = %d, grp_attn_n = %ld, grp_attn_w = %ld\n", n_ctx_train, static_cast<long>(ga_n), static_cast<long>(ga_w));
    }
    
    if (params.interactive) {
        const char * control_message;
        if (params.multilineInput) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
            " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
            " - To return control without starting a new line, end your input with '/'.\n"
            " - If you want to submit another line, end your input with '\\'.\n";
        }
        
        isInteracting = params.interactiveFirst;
    }
    
    is_antiprompt        = false;
    input_echo           = true;
    display              = true;
    need_to_save_session = [pathSession length] > 0 && n_matching_session_tokens < embd_inp.size();
    n_remain           = params.nPredict;
    
    antiprompt_ids.reserve([params.antiPrompts count]);
    for (NSString *antiprompt in params.antiPrompts) {
        antiprompt_ids.emplace_back([self.ctx tokenize:antiprompt addSpecial:false parseSpecial:true]);
    }
    
    if ([self.model hasEncoder]) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();
        
        if ([_ctx encode:llama_batch_get_one(enc_input_buf, enc_input_size)]) {
            [NSException raise:@"EvalFailure" format:@"failed to eval"];
        }
        
        llama_token decoder_start_token_id = llama_model_decoder_start_token([self.model cModel]);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = [self.model tokenBOS];
        }
        
        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }
    return self;
}

// MARK: LastOutput
- (NSString *)lastOutput {
    return [_mutableLastOutput copy];
}

- (void)start {
    while ((n_remain != 0 && !is_antiprompt) || _params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                os_log_error(os_log_inst, "<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
            }

            if (_params.grpAttnN == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= [_ctx nCtx]) {
                    if (!_params.ctxShift) {
                        os_log_debug(os_log_inst, "\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    } else {
                        if (_params.nPredict == -2) {
                            os_log_debug(os_log_inst, "\n\n%s: context full and n_predict == -%d => stopping\n", __func__, _params.nPredict);
                            break;
                        }

                        const int n_left    = n_past - _params.nKeep;
                        const int n_discard = n_left/2;

                        os_log_debug(os_log_inst, "context full, swapping: n_past = %d, n_left = %d, n_ctx = %lu, n_keep = %d, n_discard = %d\n",
                                     n_past, n_left, static_cast<unsigned long>([_ctx nCtx]), _params.nKeep, n_discard);

                        llama_kv_cache_seq_rm ([self.ctx cContext], 0, _params.nKeep            , _params.nKeep + n_discard);
                        llama_kv_cache_seq_add([self.ctx cContext], 0, _params.nKeep + n_discard, n_past, -n_discard);

                        n_past -= n_discard;

                        os_log_debug(os_log_inst, "after swap: n_past = %d\n", n_past);

                        os_log_debug(os_log_inst, "embd: %s\n", [self.ctx convertTokensToString:embd].c_str());

                        os_log_debug(os_log_inst, "clear session path\n");
                        [pathSession setString:@""];
                    }
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    os_log_debug(os_log_inst, "\n");
                    os_log_debug(os_log_inst, "shift: [%6ld, %6d] + %6d -> [%6ld, %6d]\n", static_cast<long>(ga_i), n_past, ib*bd, static_cast<long>(ga_i + ib*bd), n_past + ib*bd);
                    os_log_debug(os_log_inst, "div:   [%6ld, %6ld] / %6ld -> [%6ld, %6ld]\n", static_cast<long>(ga_i + ib*bd), static_cast<long>(ga_i + ib*bd + ga_w), static_cast<long>(ga_n), static_cast<long>((ga_i + ib*bd)/ga_n), static_cast<long>((ga_i + ib*bd + ga_w)/ga_n));
                    os_log_debug(os_log_inst, "shift: [%6ld, %6d] + %6d -> [%6ld, %6d]\n", static_cast<long>(ga_i + ib*bd + ga_w), n_past + ib*bd, dd, static_cast<long>(ga_i + ib*bd + ga_w + dd), n_past + ib*bd + dd);

                    [self.ctx kvCacheSeqAdd:0 p0:ga_i p1:n_past delta:ib*bd];
                    [self.ctx kvCacheSeqDiv:0 p0:ga_i + ib*bd p1:ga_i + ib*bd + ga_w delta:ga_n];
                    [self.ctx kvCacheSeqAdd:0 p0:ga_i + ib*bd + ga_w p1:n_past + ib*bd delta:dd];

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    os_log_debug(os_log_inst, "\nn_past_old = %d, n_past = %d, ga_i = %ld\n\n", n_past + bd, n_past, static_cast<long>(ga_i));
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += _params.nBatch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > _params.nBatch) {
                    n_eval = _params.nBatch;
                }

                os_log_debug(os_log_inst, "eval: %s\n", [self.ctx convertTokensToString:embd].c_str());

                
                if ([self.ctx decode:[[LlamaBatch alloc] initWithBatch:llama_batch_get_one(&embd[i], n_eval)] ]) {
                    [NSException raise:@"EvalFailure" format:@"failed to eval"];
                }

                n_past += n_eval;

                os_log_debug(os_log_inst, "n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (_params.nPrint > 0 && n_past % _params.nPrint == 0) {
                    os_log_debug(os_log_inst, "\n\033[31mTokens consumed so far = %d / %lu \033[0m\n", n_past, static_cast<unsigned long>([self.ctx nCtx]));
                }
            }

            if (!embd.empty() && [pathSession length] > 0) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !isInteracting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if ([pathSession length] > 0 && need_to_save_session && !_params.promptCacheRO) {
                need_to_save_session = false;
                [self.ctx saveStateFile:pathSession
                                 tokens:session_tokens.data()
                            nTokenCount:session_tokens.size()];

                os_log_debug(os_log_inst, "saved session to %s\n", [pathSession cStringUsingEncoding:NSUTF8StringEncoding]);
            }

            const llama_token idToken = [_smpl sample:self.ctx index:-1];

            [_smpl accept:idToken acceptGrammar:true];

            embd.push_back(idToken);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            os_log_debug(os_log_inst, "n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            os_log_debug(os_log_inst, "embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                [_smpl accept:embd_inp[n_consumed] acceptGrammar:false];

                ++n_consumed;
                if ((int) embd.size() >= _params.nBatch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            
            for (auto idToken : embd) {
                NSString *token_str = [self.ctx tokenToPiece:idToken special:_params.special];

                // Console/Stream Output
                os_log_info(os_log_inst, "%s", [token_str cStringUsingEncoding:NSUTF8StringEncoding]);

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(idToken);
                    
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(idToken);
                    output_ss << [token_str cStringUsingEncoding:NSUTF8StringEncoding];
                    last_output_ss << [token_str cStringUsingEncoding:NSUTF8StringEncoding];
                    [self willChangeValueForKey:@"lastOutput"];
                    [_mutableLastOutput appendString:token_str];
                    [self didChangeValueForKey:@"lastOutput"];
                }
                
            }
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if ([_params.antiPrompts count] > 0) {
                const int n_prev = 32;
                NSString *last_output = [_smpl previousString:self.ctx n:n_prev];

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (NSString *antiprompt in _params.antiPrompts) {
                    size_t extra_padding = _params.interactive ? 0 : 2;
                    size_t search_start_pos = [last_output length] > static_cast<size_t>([antiprompt length] + extra_padding)
                    ? [last_output length] - static_cast<size_t>([antiprompt length] + extra_padding)
                        : 0;

                    // TODO: Check if correct
                    if ([last_output rangeOfString:antiprompt options:0 range:NSMakeRange(search_start_pos, last_output.length - search_start_pos)].location != NSNotFound) {
                        if (_params.interactive) {
                            isInteracting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                llama_token last_token = [_smpl last];
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        if (_params.interactive) {
                            isInteracting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    os_log_debug(os_log_inst, "found antiprompt: %s\n", [last_output cStringUsingEncoding:NSUTF8StringEncoding]);
                }
            }

            // deal with end of generation tokens in interactive mode
            
            if ([self.model tokenIsEOG:[_smpl last]]) {
                os_log_debug(os_log_inst, "found an EOG token\n");

                if (_params.interactive) {
                    if ([[_params antiPrompts] count] > 0) {
                        // tokenize and inject first reverse prompt
                        
                        const auto first_antiprompt = [self.ctx tokenize:_params.antiPrompts[0] addSpecial:false parseSpecial:true];
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (_params.enableChatTemplate) {
                        [self chat_add_and_format:chat_msgs
                                             role:"assistant"
                                          content:assistant_ss.str()];
                    }
                    isInteracting = true;
                }
            }

            // if current token is not EOG, we add it to current assistant message
            if (_params.conversation) {
                const auto idToken = [_smpl last];
                assistant_ss << [[self.ctx tokenToPiece:idToken special:false] cStringUsingEncoding:NSUTF8StringEncoding];
            }

            if (n_past > 0 && isInteracting) {
                os_log_debug(os_log_inst, "waiting for user input\n");

                if (_params.conversation) {
//                    osLog_("\n> ");
                }

                if (_params.inputPrefixBOS) {
                    os_log_debug(os_log_inst, "adding input prefix BOS token\n");
                    embd_inp.push_back([self.model tokenBOS]);
                }

                std::string buffer;
                if ([_params.inputPrefix length] > 0 && !_params.conversation) {
                    os_log_debug(os_log_inst, "appending input prefix: '%s'\n", [_params.inputPrefix cStringUsingEncoding:NSUTF8StringEncoding]);
                    os_log_info(os_log_inst, "%s", [_params.inputPrefix cStringUsingEncoding:NSUTF8StringEncoding]);
                }

                display = _params.displayPrompt;
                
                if (!last_output_ss.str().empty()) {
                    auto str = last_output_ss.str();
                    last_output_ss.str("");
                    [_queue addOutputLine:[NSString stringWithCString:str.c_str() encoding:NSUTF8StringEncoding]];
                    [self willChangeValueForKey:@"lastOutput"];
                    _mutableLastOutput = [[NSMutableString alloc] init];
                    [self didChangeValueForKey:@"lastOutput"];
                    
                }
                
                buffer = [[_queue inputLine] cStringUsingEncoding:NSUTF8StringEncoding];
                if ([_queue isClosed]) {
                    break;
                }
                
                display = true;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if ([[self params].inputSuffix length] > 0 && !_params.conversation) {
                        os_log_debug(os_log_inst, "appending input suffix: '%s'\n", [_params.inputSuffix cStringUsingEncoding:NSUTF8StringEncoding]);
                        os_log_info(os_log_inst, "%s", [_params.inputSuffix cStringUsingEncoding:NSUTF8StringEncoding]);
                    }

                    os_log_debug(os_log_inst, "buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (_params.escapeSequences) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = _params.conversation && _params.enableChatTemplate;
                    std::string user_inp = format_chat
                    ? [[self chat_add_and_format:chat_msgs role:"user" content:std::move(buffer)] cStringUsingEncoding:NSUTF8StringEncoding]
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = [self.ctx tokenize:_params.inputPrefix addSpecial:false parseSpecial:true];
                    const auto line_inp = [self.ctx tokenize:[NSString stringWithCString:user_inp.c_str()
                                                                           encoding:NSUTF8StringEncoding]
                                             addSpecial:false
                                           parseSpecial:format_chat];
                    const auto line_sfx = [self.ctx tokenize:_params.inputSuffix
                                             addSpecial:false
                                           parseSpecial:true];

                    os_log_debug(os_log_inst, "input tokens: %s\n", [self.ctx convertTokensToString:line_inp].c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = [self.model tokenEOT];
                        embd_inp.push_back(eot == -1 ? [self.model tokenEOS] : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << [[self.ctx tokenToPiece:token] cStringUsingEncoding:NSUTF8StringEncoding];
                    }
                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    os_log_debug(os_log_inst, "n_remain: %d\n", n_remain);
                } else {
                    os_log_debug(os_log_inst, "empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (isInteracting) {
                    [_smpl reset];
                }
                isInteracting = false;
            }
        }

        // end of generation
        if (!embd.empty() && [self.model tokenIsEOG:embd.back()] && !(_params.interactive)) {
            os_log_info(os_log_inst, " [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (_params.interactive && n_remain <= 0 && _params.nPredict >= 0) {
            n_remain = _params.nPredict;
            isInteracting = true;
        }
    }
    
    NSLog(@"Loop over");
    [_queue addOutputLine:@""];
}

- (void)stop {
    isInteracting = false;
    _params.interactive = false;
    _queue.isClosed = YES;
    [_queue addInputLine:@""];
    [_queue outputLine];
}

- (void)dealloc
{
    [_queue dealloc];
    [self.smpl dealloc];
    [self.ctx dealloc];
    [self.model dealloc];
    llama_backend_free();
    [threadpool dealloc];
    [threadpool_batch dealloc];
    
    [super dealloc];
}

@end
