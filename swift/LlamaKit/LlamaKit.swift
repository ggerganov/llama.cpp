import Foundation
@_exported import JSONSchema
@_exported import LlamaObjC

public protocol DynamicCallable: Sendable {
    @discardableResult
    func dynamicallyCall(withKeywordArguments args: [String: Any]) async throws -> String
}


struct ToolCall: Decodable {
    let id: Int
    let name: String
    let arguments: [String: String]
}

struct ToolResponse<T: Encodable>: Encodable {
    let id: Int
    let result: T
}

// MARK: LlamaChatSession
/// Standard chat session for a given LLM.
public actor LlamaChatSession {
    private let queue = BlockingLineQueue()
    private let session: LlamaObjC.LlamaSession
    
    public init(params: GPTParams, flush: Bool = true) async throws {
        session = LlamaObjC.LlamaSession(params: params)
        Task.detached { [session, queue] in
            session.start(queue)
        }
        
        // flush
        guard flush else { return }
        _ = queue.outputLine()
    }
    
    public func chat(message: String) async -> String {
        queue.addInputLine(message)
        return queue.outputLine()
    }
}

// MARK: LlamaGrammarSession
public actor LlamaSession<T: JSONSchemaConvertible> {
    private let session: LlamaChatSession
    
    public init(params: GPTParams) async throws {
        let converter = SchemaConverter(propOrder: [])
        _ = converter.visit(schema: T.jsonSchema, name: nil)
        params.samplerParams.grammar = converter.formatGrammar()
        session = try await LlamaChatSession(params: params)
    }
    
    public func chat(message: String) async throws -> T {
        let output = await session.chat(message: message).data(using: .utf8)!
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
        You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
        <tool_call>
        {"name": <function-name>,"arguments": <args-dict>}
        </tool_call>

        The first call you will be asked to warm up is to get the user's IP address. Here are the available tools:
        <tools> \(String(data: encoded, encoding: .utf8)!) </tools><|eot_id|>
        """
        params.prompt = prompt
        params.interactive = true
        params.antiPrompts.append("<|eot_id|>");
        params.inputPrefix = "<|start_header_id|>user<|end_header_id|>";
        params.inputSuffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        session = try await LlamaChatSession(params: params, flush: false)
        let fn = await session.chat(message: "What is my IP address?")
        let toolCall = try JSONDecoder().decode(ToolCall.self, from: fn.data(using: .utf8)!)
        guard let tool = self.tools[toolCall.name] else {
            fatalError()
        }
        let resp = try await tool.0.dynamicallyCall(withKeywordArguments: toolCall.arguments)
        print(resp)

        let output = await session.chat(message: """
        <tool_response>
        {"id": \(toolCall.id), result: \(resp)}
        </tool_response>
        """)
        print(output)
    }
    
    public func chat(message: String) async throws -> String {
        var nxt = await session.chat(message: message)
        let fn = nxt
        // try to see if the output is a function call
        do {
            let toolCall = try JSONDecoder().decode(ToolCall.self, from: fn.data(using: .utf8)!)
            guard let tool = tools[toolCall.name] else {
                fatalError()
            }
            let callable = tool.0
            let resp = try await callable.dynamicallyCall(withKeywordArguments: toolCall.arguments)
            print("tool response: \(resp)")
            nxt = await session.chat(message: """
            <tool_response>
            {"id": \(toolCall.id), result: \(resp)}
            </tool_response>
            """)
            print(nxt)
        } catch {
            print(error)
        }
        return nxt
    }
}

public protocol LlamaActor: Actor {
    static var tools: [String: (DynamicCallable, _JSONFunctionSchema)] { get }
    var session: LlamaToolSession { get }
}

public extension LlamaActor {
    func chat(_ message: String) async throws -> String {
        try await session.chat(message: message)
    }
}

@attached(member, names: arbitrary)
@attached(extension, conformances: LlamaActor, names: arbitrary)
public macro llamaActor() = #externalMacro(module: "LlamaKitMacros",
                                           type: "LlamaActorMacro")

@attached(body)
public macro Tool() = #externalMacro(module: "LlamaKitMacros",
                                     type: "ToolMacro")
