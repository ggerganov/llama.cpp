import Foundation
@_exported import JSONSchema
@_exported import LlamaObjC

// MARK: LlamaChatSession

/// An actor that manages a standard chat session with a given Large Language Model (LLM).
///
/// `LlamaChatSession` provides methods to interact with the LLM, allowing for synchronous
/// and asynchronous message inference. It uses a `BlockingLineQueue` to manage the input
/// and output lines for communication with the LLM backend.
///
/// ### Key Responsibilities:
/// - Initialize the session with the specified parameters.
/// - Send messages to the LLM and receive responses.
/// - Provide an asynchronous stream for real-time inference results.
public actor LlamaChatSession {
    /// The underlying session object interfacing with the LLM backend.
    internal let session: __LlamaSession
    
    /// Initialize the session
    /// - parameter params: common parameters to initialize the session
    /// - parameter flush: whether or not to flush the initial prompt, reading initial output
    public init(params: GPTParams, flush: Bool = true) async throws {
        self.session = __LlamaSession(params: params)
        Task.detached { [session] in
            session.start()
        }
        
        // flush
        guard flush else { return }
        _ = session.queue.outputLine()
    }
    
    /// Creates an asynchronous stream for inference results based on a given message.
    ///
    /// This method sends a message to the LLM and returns an `AsyncStream` that yields
    /// output as it becomes available, allowing for real-time streaming of inference results.
    ///
    /// - Parameter message: The message to send to the LLM for inference.
    /// - Returns: An `AsyncStream` of `String` values representing incremental output from the LLM.
    public func inferenceStream(message: String) async -> AsyncStream<String> {
        session.queue.addInputLine(message)
        var observationToken: NSKeyValueObservation?
        return AsyncStream { stream in
            observationToken = self.session.observe(\.lastOutput, options: [.new, .old]) { session, change in
                guard let newValue = change.newValue,
                      let oldValue = change.oldValue else {
                    return stream.finish()
                }
                var delta = ""
                for change in newValue!.difference(from: oldValue!) {
                    switch change {
                    case .remove(_, _, _):
                        return stream.finish()
                    case .insert(_, let element, _):
                        delta.append(element)
                    }
                }
                stream.yield(delta)
            }
            stream.onTermination = { [observationToken] _ in
                observationToken?.invalidate()
            }
        }
    }

    /// Sends a message to the LLM and returns the response.
    ///
    /// This method sends a message to the LLM and waits for the complete response.
    ///
    /// - Parameter message: The message to send to the LLM for inference.
    /// - Returns: A `String` containing the LLM's response to the message.
    public func infer(message: String) async -> String {
        session.queue.addInputLine(message)
        return session.queue.outputLine()
    }
    
    deinit {
        session.stop()
    }
}


// MARK: LlamaGrammarSession

/// An actor that manages a chat session with the LLM, enforcing a grammar defined by a JSON schema.
///
/// `LlamaSession` allows you to interact with the LLM while constraining the output to match a specified JSON schema.
/// It uses the `JSONSchema` and `JSONSchemaConvertible` protocols to define the expected output format,
/// and applies a grammar to the LLM's sampler parameters.
///
/// ### Key Responsibilities:
/// - Initialize the session with a specified JSON schema, converting it to a grammar for the LLM.
/// - Send messages to the LLM and decode the responses into a specified type `T`.
///
/// ### Type Parameters:
/// - `T`: A type that conforms to `JSONSchemaConvertible` and represents the expected output format.
public actor LlamaSession<T: JSONSchemaConvertible> {
    /// The underlying chat session used to interact with the LLM.
    private let session: LlamaChatSession
    
    /// Initializes a new grammar-constrained session with the LLM.
    ///
    /// - Parameters:
    ///   - params: The parameters used to initialize the session, such as model settings.
    ///   - flush: A boolean indicating whether to flush the initial prompt and read initial output.
    ///            Defaults to `true`.
    /// - Throws: An error if the session fails to initialize.
    public init(params: GPTParams, flush: Bool = true) async throws {
        let converter = SchemaConverter(propOrder: [])
        _ = converter.visit(schema: T.jsonSchema, name: nil)
        params.samplerParams.grammar = converter.formatGrammar()
        session = try await LlamaChatSession(params: params, flush: flush)
    }
    
    /// Sends a message to the LLM and decodes the response into type `T`.
    ///
    /// This method sends a message to the LLM, receives the output, and attempts to decode it into the specified type `T`.
    /// It enforces that the LLM's output conforms to the JSON schema associated with `T`.
    ///
    /// - Parameter message: The message to send to the LLM.
    /// - Returns: An instance of type `T` decoded from the LLM's response.
    /// - Throws: An error if the decoding fails.
    public func infer(message: String) async throws -> T {
        let output = await session.infer(message: message).data(using: .utf8)!
        return try JSONDecoder().decode(T.self, from: output)
    }
}
