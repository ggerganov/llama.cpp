import Foundation

// import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    
    private var tokens_list: [llama_token]
    
    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.tokens_list = []
    }
    
    deinit {
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }
    
    static func createContext(path: String) throws -> LlamaContext {
        llama_backend_init(false)
        let params = llama_context_default_params()
        let model = llama_load_model_from_file(path, params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }
        
        let context = llama_new_context_with_model(model, params)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }
        
        return LlamaContext(model: model, context: context)
    }
    
    func get_kv_cache() -> Int32 {
        return llama_get_kv_cache_token_count(context)
    }
    
    func completion_init(text: String) -> Int32 {
        print("attempting to complete \(text)...")
        
        tokens_list = tokenize(text: text, add_bos: true)
        
        let max_context_size = llama_n_ctx(context)
        let max_tokens_list_size = max_context_size - 4
        
        if tokens_list.count > max_tokens_list_size {
            print("error: prompt too long (\(tokens_list.count) tokens, max \(max_tokens_list_size)")
        }
        
        for id in tokens_list {
            print(token_to_piece(token: id))
        }
        
        let n_gen = min(32, max_context_size)
        return n_gen
    }
    
    func completion_loop() -> String {
        var done = false
        tokens_list.withUnsafeBufferPointer() { cArray in
            let res = llama_eval(context, cArray.baseAddress, Int32(tokens_list.count), llama_get_kv_cache_token_count(context), 8)
            if res != 0 {
                print("error evaluating llama!")
                done = true
                return
            }
        }
        if done {
            return ""
        }
        
        tokens_list.removeAll()
        
        var new_token_id: llama_token = 0
        
        let logits = llama_get_logits(context)
        let n_vocab = llama_n_vocab(context)
        
        var candidates = Array<llama_token_data>()
        candidates.reserveCapacity(Int(n_vocab))
        
        for token_id in 0...n_vocab {
            candidates.append(llama_token_data(id: token_id, logit: logits![Int(token_id)], p: 0.0))
        }
        candidates.withUnsafeMutableBufferPointer() { buffer in
            var candidates_p = llama_token_data_array(data: buffer.baseAddress, size: buffer.count, sorted: false)
            
            new_token_id = llama_sample_token_greedy(context, &candidates_p)
        }
        let new_token_str = token_to_piece(token: new_token_id)
        print(new_token_str)
        tokens_list.append(new_token_id)
        return new_token_str
    }
    
    func clear() {
        tokens_list.removeAll()
    }
    
    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let n_tokens = text.count + (add_bos ? 1 : 0)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(context, text, Int32(text.count), tokens, Int32(n_tokens), add_bos)
        
        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }
        
        tokens.deallocate()
        
        return swiftTokens
    }
    
    private func token_to_piece(token: llama_token) -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        
        let _ = llama_token_to_piece(context, token, result, 8)
        
        let resultStr = String(cString: result)
        
        result.deallocate()
        
        return resultStr
    }
}
