import Foundation


public protocol DynamicCallable: Sendable {
    @discardableResult
    func dynamicallyCall(withKeywordArguments args: [String: Any]) async throws -> String
}

public enum AnyDecodable: Decodable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    // Add other cases as needed

    // Initializers for each type
    init(_ value: String) {
        self = .string(value)
    }

    init(_ value: Int) {
        self = .int(value)
    }

    init(_ value: Double) {
        self = .double(value)
    }

    init(_ value: Bool) {
        self = .bool(value)
    }

    init() {
        self = .null
    }

    // Decodable conformance
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if container.decodeNil() {
            self = .null
        } else if let intValue = try? container.decode(Int.self) {
            self = .int(intValue)
        } else if let doubleValue = try? container.decode(Double.self) {
            self = .double(doubleValue)
        } else if let boolValue = try? container.decode(Bool.self) {
            self = .bool(boolValue)
        } else if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
        } else {
            let context = DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "Cannot decode AnyDecodable"
            )
            throw DecodingError.typeMismatch(AnyDecodable.self, context)
        }
    }
}

struct ToolCall: Decodable {
    let id: Int
    let name: String
    let arguments: [String: AnyDecodable]
}

struct ToolResponse<T: Encodable>: Encodable {
    let id: Int
    let result: T
}

// MARK: LlamaToolSession
/// An actor that manages tool calls within a chat session using Llama language models.
///
/// `LlamaToolSession` is responsible for handling interactions between the language model and dynamically callable tools.
/// It maintains a dictionary of tools that can be invoked based on the language model's output.
/// With the integration of the `@llamaActor` and `@Tool` macros, users can easily define custom tools within their own actors,
/// which are then registered and managed by `LlamaToolSession`.
///
/// ### Key Responsibilities:
/// - Initializes the language model session with the provided tools and prompt.
/// - Parses the language model's output to detect tool calls and invokes the corresponding tools.
/// - Sends tool responses back to the language model to generate coherent replies.
/// - Supports both synchronous and asynchronous inference, including streaming results.
///
/// ### Integration with Macros:
/// - The `@llamaActor` macro simplifies the creation of actors with tools by automatically generating the necessary code to integrate with `LlamaToolSession`.
/// - The `@Tool` macro allows functions within the actor to be marked as tools, which are registered and made available to the language model.
///
/// ### Usage:
/// ```swift
/// @llamaActor actor MyLlama {
///     /// Gets the user's favorite season.
///     @Tool public func getFavoriteSeason() async throws -> String {
///         return "autumn"
///     }
///
///     /// Gets the user's favorite animal.
///     @Tool public func getFavoriteAnimal() async throws -> String {
///         return "cat"
///     }
/// }
///
/// let params = GPTParams(prompt: "Your initial prompt")
/// let myLlama = try await MyLlama(params: params)
/// let response = try await myLlama.session.infer(message: "What is my favorite season?")
/// print(response)
/// ```
///
/// ### Architecture Diagram:
/// ```
/// +-------------------+
/// |   MyLlama Actor   |
/// |  (User-Defined)   |
/// |                   |
/// | @llamaActor       |
/// | +---------------+ |
/// | | getFavorite...|<------------------------------+
/// | | (Tool funcs)  |                              |
/// +---------+-------+                              |
///           |                                      |
///           | Uses                                 |
///           v                                      |
/// +-------------------+          +-----------------+       +------------------+
/// | LlamaToolSession  |          | LlamaChatSession|       | __LlamaSession   |
/// |       (Actor)     |          |     (Actor)     |       | (Objective-C)    |
/// +---------+---------+          +---------+-------+       +---------+--------+
///           |                               ^                       ^
///           |                               |                       |
///           |                               | Observes              |
///           v                               |                       |
/// +-------------------+          +-----------------+       +------------------+
/// | BlockingLineQueue |---------->| LlamaSession   |<------|  Tools (Macros)  |
/// |   (Objective-C)   |   Input/  | (Objective-C)  |       | (DynamicCallable)|
/// +-------------------+   Output  +----------------+       +------------------+
/// ```
///
/// ### Notes:
/// - **Beta Feature**: This feature is currently in beta and may undergo significant changes.
/// - **Template Format**: `LlamaToolSession` currently only works with the llama chat template format.
/// - **Thread Safety**: `LlamaToolSession` is an actor to ensure thread-safe operations in a concurrent environment.
/// - **Extensibility**: Users can define custom tools using the `@Tool` macro within an actor annotated with `@llamaActor`.
/// - **Error Handling**: The session handles tool call failures gracefully, allowing the language model to continue processing.
///
/// ### See Also:
/// - `LlamaChatSession`
/// - `@llamaActor` Macro
/// - `@Tool` Macro
public actor LlamaToolSession {
    private let session: LlamaChatSession
    
    public private(set) var tools: [String: (DynamicCallable, _JSONFunctionSchema)]
    
    public init(params: GPTParams,
                tools: [String: (DynamicCallable, _JSONFunctionSchema)]) async throws {
        self.tools = tools
        let encoded = try JSONEncoder().encode(self.tools.values.map(\.1))
        let prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        \(params.prompt ?? "")

        You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
        <tool_call>
        {"name": <function-name>,"arguments": <args-dict>}
        </tool_call>

        Feel free to chain tool calls, e.g., if you need the user's location to find points of interest near them, fetch the user's location first.

        <tools> \(String(data: encoded, encoding: .utf8)!) </tools>
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        params.prompt = prompt
        params.interactive = true
        params.antiPrompts.append("<|eot_id|>")
        params.inputPrefix = "<|start_header_id|>user<|end_header_id|>"
        params.inputSuffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        session = try await LlamaChatSession(params: params, flush: false)
        guard let line = await session.session.queue.outputLine() else {
            return
        }
        _ = await callTool(line)
    }
    
    private func callTool(_ call: ToolCall) async throws -> String {
        guard let tool = tools[call.name] else {
            fatalError()
        }
        // TODO: tool call decode is allowed to fail but the code below is not
        let callable = tool.0
    
        return try await callable.dynamicallyCall(withKeywordArguments: call.arguments)
    }

    private func callTool(_ call: String) async -> String? {
        var nxt: String?
        do {
            let toolCall = try JSONDecoder().decode(ToolCall.self, from: call.data(using: .utf8)!)
            guard let tool = tools[toolCall.name] else {
                fatalError()
            }
            // TODO: tool call decode is allowed to fail but the code below is not
            let callable = tool.0
        
            do {
                let response = try await callable.dynamicallyCall(withKeywordArguments: toolCall.arguments)
                print("tool response: \(response)")
                nxt = await session.infer(message: """
                <tool_response>
                {"id": \(toolCall.id), result: \(response)}
                </tool_response>
                """)
                // TODO: If this decodes correctly, we should tail this into this method
                // TODO: so that we do not decode twice
                if let _ = try? JSONDecoder().decode(ToolCall.self, from: nxt!.data(using: .utf8)!) {
                    return await callTool(nxt!)
                }
            } catch {
                nxt = await session.infer(message: """
                <tool_response>
                {"id": \(toolCall.id), result: "The tool call has unfortunately failed."}
                </tool_response>
                """)
            }
            print(nxt ?? "nil")
        } catch {}
        return nxt
    }
    
    public func infer(message: String) async throws -> String {
        let output = await session.infer(message: message)
        guard let output = await callTool(output) else {
            return output
        }
        return output
    }

    public func inferenceStream(message: String) async -> AsyncStream<String> {
        let underlyingSession = await session.session
        await session.session.queue.addInputLine(message)
        var observationToken: NSKeyValueObservation?
        actor StreamProcessor {
            private(set) var buffer = ""
            private(set) var totalBuffer = ""
            private(set) var isProcessingToolCall = false
            
            func setBuffer(_ value: String) {
                buffer = value
            }
            func appendBuffer(_ value: String) {
                buffer.append(value)
                totalBuffer.append(value)
            }
            func setTotalBuffer(_ value: String) {
                totalBuffer = value
            }
            func setIsProcessingToolCall(_ value: Bool) {
                isProcessingToolCall = value
            }
        }
        
        let streamProcessor = StreamProcessor()
        return AsyncStream { continuation in
            observationToken = underlyingSession.observe(\.lastOutput, options: [.new, .old]) { session, change in
                Task {
                    guard let newValue = change.newValue,
                          let oldValue = change.oldValue else {
                        continuation.finish()
                        return
                    }
                    var delta = ""
                    for change in newValue!.difference(from: oldValue!) {
                        switch change {
                        case .remove(_, _, _):
                            if await streamProcessor.isProcessingToolCall {
                                let call = try await JSONDecoder().decode(ToolCall.self, from: streamProcessor.totalBuffer.data(using: .utf8)!)
                                await streamProcessor.setBuffer("")
                                await streamProcessor.setTotalBuffer("")
                                let response = try await self.callTool(call)
                                return await self.session.session.queue.addInputLine("""
                                <tool_response>
                                {"id": \(call.id), result: \(response)}
                                </tool_response>
                                """)
                            } else {
                                continuation.finish()
                            }
                            return
                        case .insert(_, let element, _):
                            delta.append(element)
                        }
                    }
                    await streamProcessor.appendBuffer(delta)
                    // Check if buffer contains a complete tool call
                    if await streamProcessor.totalBuffer.starts(with: "\n\n\n{") {
                        await streamProcessor.setIsProcessingToolCall(true)
                    } else {
                        await streamProcessor.setIsProcessingToolCall(false)
                        // If not a tool call, yield the delta
                        continuation.yield(delta)
                        await streamProcessor.setBuffer("")
                    }
                    // Else, continue buffering until we have a complete tool call
                }
            }
            continuation.onTermination = { [observationToken] _ in
                observationToken?.invalidate()
            }
        }
    }
}

public protocol LlamaActor: Actor {
    static func tools(_ self: Self) -> [String: (DynamicCallable, _JSONFunctionSchema)]
    var session: LlamaToolSession! { get }
    
}

public extension LlamaActor {
    func infer(_ message: String) async throws -> String {
        try await session.infer(message: message)
    }
    func inferenceStream(message: String) async -> AsyncStream<String> {
        await session.inferenceStream(message: message)
    }
}

// MARK: @llamaActor Macro
/// A macro that transforms an actor into a Llama tool actor, automatically integrating with `LlamaToolSession`.
///
/// The `@llamaActor` macro simplifies the process of creating an actor with tools that can be used by the Llama language model.
/// It automatically generates the necessary code to initialize a `LlamaToolSession`, register the tools, and conform to the `LlamaActor` protocol.
///
/// ### Usage:
/// ```swift
/// @llamaActor actor MyLlama {
///     /// Gets the user's favorite season.
///     @Tool public func getFavoriteSeason() async throws -> String {
///         return "autumn"
///     }
///
///     /// Gets the user's favorite animal.
///     @Tool public func getFavoriteAnimal() async throws -> String {
///         return "cat"
///     }
/// }
/// ```
///
/// ### Macro Details:
/// - **Attached To**: Actor declarations.
/// - **Produces**:
///   - Member variables and initializers required for tool integration.
///   - An extension conforming the actor to `LlamaActor`, providing necessary functionalities.
/// - **Parameters**: None.
///
/// ### Notes:
/// - The macro processes functions marked with `@Tool` within the actor to generate dynamic callable tools.
/// - It collects the tools and their schemas to register them with `LlamaToolSession`.
///
/// ### See Also:
/// - `@Tool` Macro
/// - `LlamaToolSession`
@attached(member, names: arbitrary)
@attached(extension, conformances: LlamaActor, names: arbitrary)
public macro llamaActor() = #externalMacro(module: "LlamaKitMacros",
                                           type: "LlamaActorMacro")

/// A macro that marks a function within an actor as a tool callable by the Llama language model.
///
/// The `@Tool` macro indicates that a function should be exposed as a tool to the language model.
/// It processes the function to generate a dynamically callable structure, registers it with `LlamaToolSession`,
/// and includes the tool's metadata in the model's prompt.
///
/// ### Usage:
/// ```swift
/// @llamaActor actor MyLlama {
///     /// Gets the user's favorite animal.
///     @Tool public func getFavoriteAnimal() async throws -> String {
///         return "cat"
///     }
/// }
/// ```
///
/// ### Macro Details:
/// - **Attached To**: Function declarations within an actor marked with `@llamaActor`.
/// - **Produces**:
///   - A dynamically callable structure that wraps the function.
///   - Registers the tool with its name, description, and parameters.
/// - **Parameters**: None.
///
/// ### Notes:
/// - The function's documentation comment is used as the tool's description.
/// - Parameter comments are used to describe the tool's parameters.
/// - Supports functions with parameters and return values that conform to `Codable`.
///
/// ### See Also:
/// - `@llamaActor` Macro
/// - `LlamaToolSession`
@attached(body)
public macro Tool() = #externalMacro(module: "LlamaKitMacros",
                                     type: "ToolMacro")
