import Foundation
@_exported import JSONSchema
@_exported import LlamaObjC

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

// MARK: LlamaChatSession
/// Standard chat session for a given LLM.
public actor LlamaChatSession {
    private let queue = BlockingLineQueue()
    private let session: __LlamaSession
    
    /// Initialize the session
    /// - parameter params: common parameters to initialize the session
    /// - parameter flush: whether or not to flush the initial prompt, reading initial output
    public init(params: GPTParams, flush: Bool = true) async throws {
        self.session = __LlamaSession(params: params)
        Task.detached { [session, queue] in
            session.start(queue)
        }
        
        // flush
        guard flush else { return }
        _ = queue.outputLine()
    }
    
    /// Create a new inference stream for a given message
    /// - parameter message: The message to receive an inference for.
    /// - returns: A stream of output from the LLM.
    public func inferenceStream(message: String) async -> AsyncStream<String> {
        queue.addInputLine(message)
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
    
    public func infer(message: String) async -> String {
        queue.addInputLine(message)
        return queue.outputLine()
    }
}

// MARK: LlamaGrammarSession
public actor LlamaSession<T: JSONSchemaConvertible> {
    private let session: LlamaChatSession
    
    public init(params: GPTParams, flush: Bool = true) async throws {
        let converter = SchemaConverter(propOrder: [])
        _ = converter.visit(schema: T.jsonSchema, name: nil)
        params.samplerParams.grammar = converter.formatGrammar()
        session = try await LlamaChatSession(params: params, flush: flush)
    }
    
    public func chat(message: String) async throws -> T {
        let output = await session.infer(message: message).data(using: .utf8)!
        return try JSONDecoder().decode(T.self, from: output)
    }
}

// MARK: LlamaToolSession
public actor LlamaToolSession {
    private let session: LlamaChatSession
    
    private struct GetIpAddress: DynamicCallable {
        func dynamicallyCall(withKeywordArguments args: [String : Any]) async throws -> String {
            getIPAddress()
        }
    }
    
    internal static func getIPAddress() -> String {
        var address: String!

        // Get list of all interfaces on the local machine:
        var ifaddr: UnsafeMutablePointer<ifaddrs>? = nil
        if getifaddrs(&ifaddr) == 0 {
            // Loop through linked list of interfaces
            var ptr = ifaddr
            while ptr != nil {
                let interface = ptr!.pointee

                // Check if the interface is IPv4 or IPv6:
                let addrFamily = interface.ifa_addr.pointee.sa_family
                if addrFamily == UInt8(AF_INET) || addrFamily == UInt8(AF_INET6) {

                    // Convert interface name to String:
                    let name = String(cString: interface.ifa_name)
                    
                    // Only consider non-loopback interfaces (e.g., "en0" for Wi-Fi)
                    if name == "en0" {  // Typically en0 is the Wi-Fi interface
                        // Convert the address to a readable format:
                        var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                        if getnameinfo(interface.ifa_addr, socklen_t(interface.ifa_addr.pointee.sa_len),
                                       &hostname, socklen_t(hostname.count),
                                       nil, socklen_t(0), NI_NUMERICHOST) == 0 {
                            address = String(cString: hostname)
                        }
                    }
                }

                ptr = interface.ifa_next
            }

            freeifaddrs(ifaddr)
        }

        return address
    }
    
    public private(set) var tools: [String: (DynamicCallable, _JSONFunctionSchema)]
    
    public init(params: GPTParams,
                tools: [String: (DynamicCallable, _JSONFunctionSchema)]) async throws {
        self.tools = tools
        let ipFnSchema = _JSONFunctionSchema(name: "getIpAddress", description: "Get the IP Address for this system", parameters: _JSONFunctionSchema.Parameters(properties: [:], required: []))
        self.tools["getIpAddress"] = (GetIpAddress(), ipFnSchema)
        let encoded = try JSONEncoder().encode(self.tools.values.map(\.1))
        let prompt = """
        \(params.prompt ?? "")

        You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
        <tool_call>
        {"name": <function-name>,"arguments": <args-dict>}
        </tool_call>

        Feel free to chain tool calls, e.g., if you need the user's location to find points of interest near them, fetch the user's location first.
        The first call you will be asked to warm up is to get the user's IP address. Here are the available tools:
        <tools> \(String(data: encoded, encoding: .utf8)!) </tools><|eot_id|>
        """
        params.prompt = prompt
        params.interactive = true
        params.antiPrompts.append("<|eot_id|>");
        params.inputPrefix = "<|start_header_id|>user<|end_header_id|>";
        params.inputSuffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        session = try await LlamaChatSession(params: params, flush: false)
        let fn = await session.infer(message: "What is my IP address?")
        let toolCall = try JSONDecoder().decode(ToolCall.self, from: fn.data(using: .utf8)!)
        guard let tool = self.tools[toolCall.name] else {
            fatalError()
        }
        let resp = try await tool.0.dynamicallyCall(withKeywordArguments: toolCall.arguments)
        print(resp)

        let output = await session.infer(message: """
        <tool_response>
        {"id": \(toolCall.id), result: \(resp)}
        </tool_response>
        """)
        print(output)
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
}

public protocol LlamaActor: Actor {
    static func tools(_ self: Self) -> [String: (DynamicCallable, _JSONFunctionSchema)]
    var session: LlamaToolSession! { get }
}

public extension LlamaActor {
    func chat(_ message: String) async throws -> String {
        try await session.infer(message: message)
    }
}

@attached(member, names: arbitrary)
@attached(extension, conformances: LlamaActor, names: arbitrary)
public macro llamaActor() = #externalMacro(module: "LlamaKitMacros",
                                           type: "LlamaActorMacro")

@attached(body)
public macro Tool() = #externalMacro(module: "LlamaKitMacros",
                                     type: "ToolMacro")
