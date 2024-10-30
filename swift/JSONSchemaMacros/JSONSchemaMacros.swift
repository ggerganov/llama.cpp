import SwiftSyntaxMacros
import SwiftCompilerPlugin
import SwiftSyntax

private struct MemberView {
    let name: String
    let type: String
    var attributeKey: String?
    var assignment: String?
}

private func view(for member: MemberBlockItemListSyntax.Element) throws -> MemberView? {
    guard let decl = member.decl.as(VariableDeclSyntax.self),
          let binding = decl.bindings.compactMap({
              $0.pattern.as(IdentifierPatternSyntax.self)
          }).first,
          let type = decl.bindings.compactMap({
              $0.typeAnnotation?.type
          }).first,
          !(type.syntaxNodeType is StructDeclSyntax.Type) else {
        return nil
    }
    var memberView = MemberView(name: "\(binding.identifier)", type: "\(type)", attributeKey: nil)
    if let macroName = decl.attributes.first?.as(AttributeSyntax.self)?
        .arguments?.as(LabeledExprListSyntax.self)?.first?.expression.as(StringLiteralExprSyntax.self) {
        memberView.attributeKey = "\(macroName.segments)"
    }
    if let assignment = decl.bindings.compactMap({
        $0.initializer?.value
    }).first {
        memberView.assignment = "\(assignment)"
    }
    return memberView
}

struct JSONSchemaMacro: ExtensionMacro, MemberMacro {
    static func expansion(of node: AttributeSyntax, providingMembersOf declaration: some DeclGroupSyntax, conformingTo protocols: [TypeSyntax], in context: some MacroExpansionContext) throws -> [DeclSyntax] {
        let members = try declaration.memberBlock.members.compactMap(view(for:))
        if declaration is EnumDeclSyntax {
            return []
        }
        return [
            """
            enum CodingKeys: CodingKey {
                case \(raw: members.map(\.name).joined(separator: ", "))
            }
            """,
            """
            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                \(raw: members.map {
                    """
                    self.\($0.name) = try \($0.type).decode(from: container, forKey: .\($0.name))
                    """
                }.joined(separator: "\n"))
            }
            """
        ]
    }
    
    static func expansion(of node: SwiftSyntax.AttributeSyntax,
                          attachedTo declaration: some SwiftSyntax.DeclGroupSyntax,
                          providingExtensionsOf type: some SwiftSyntax.TypeSyntaxProtocol,
                          conformingTo protocols: [SwiftSyntax.TypeSyntax], 
                          in context: some SwiftSyntaxMacros.MacroExpansionContext) throws -> [SwiftSyntax.ExtensionDeclSyntax] {
        let members = try declaration.memberBlock.members.compactMap(view(for:))
        var inheritedTypes: [InheritedTypeSyntax] = []
        inheritedTypes.append(InheritedTypeSyntax(type: TypeSyntax("JSONSchemaConvertible")))
        if declaration is EnumDeclSyntax {
            inheritedTypes.append(InheritedTypeSyntax(type: TypeSyntax(", CaseIterable")))
        }
        let properties = members.map {
            """
            "\($0.name)": \($0.type).jsonSchema
            """
        }
        if !(declaration is EnumDeclSyntax) {
            return [
                ExtensionDeclSyntax(extendedType: type,
                                    inheritanceClause: .init(inheritedTypes: .init(inheritedTypes)),
                                    memberBlock: """
                                {
                                    static var type: String {
                                        "object"
                                    }
                                    static var jsonSchema: [String: Any] {
                                        [
                                            "type": "object",
                                            "properties": [
                                                \(raw: properties.joined(separator: ","))
                                            ]
                                        ]
                                    }
                                }
                                """)
            ]
        } else {
            return [
                ExtensionDeclSyntax(extendedType: type,
                                    inheritanceClause: .init(inheritedTypes: .init(inheritedTypes)),
                                    memberBlock: """
                                {
                                    public static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>, forKey key: K) throws -> Self {
                                        if RawValue.self is Int.Type {
                                            return Self(rawValue: Int(try container.decode(String.self, forKey: key)) as! Self.RawValue)!
                                        } else {
                                            return try container.decode(Self.self, forKey: key)
                                        }
                                    }
                                }
                                """)
            ]
        }
    }
}

enum TestError: Error {
    case message(String)
}

struct LlamaActorMacro: ExtensionMacro, MemberMacro {
    static func expansion(of node: AttributeSyntax, providingMembersOf declaration: some DeclGroupSyntax, conformingTo protocols: [TypeSyntax], in context: some MacroExpansionContext) throws -> [DeclSyntax] {
        [
            """
            let session: LlamaToolSession
            
            public init(params: GPTParams) async throws {
                self.session = try await LlamaToolSession(params: params, tools: Self.tools)
            }
            """
        ]
    }
    
    static func expansion(of node: AttributeSyntax,
                          attachedTo declaration: some DeclGroupSyntax,
                          providingExtensionsOf type: some TypeSyntaxProtocol,
                          conformingTo protocols: [TypeSyntax],
                          in context: some MacroExpansionContext) throws -> [ExtensionDeclSyntax] {
        var tools: [
            (name: String,
             description: String,
             parameters: [(name: String,
                           type: String,
                           description: String)],
             callableString: String,
             callableName: String)
        ] = []
        for member in declaration.memberBlock.members {
            let comments = member.leadingTrivia.filter { $0.isComment }
            
            guard let member = member.decl.as(FunctionDeclSyntax.self) else {
                continue
            }
            let name = member.name
            guard case var .docLineComment(description) = comments.first else {
                throw TestError.message("Missing comment")
            }
            description = String(description.dropFirst(3))
            var parameters: [(name: String, type: String, description: String)] = []
            var index = 0
            for parameter in member.signature.parameterClause.parameters {
                let firstName = parameter.firstName.text
                let typeName = parameter.type.as(IdentifierTypeSyntax.self)!.name.text
                guard case var .docLineComment(description) = comments[index + 1] else {
                    throw TestError.message("Missing comment for \(firstName)")
                }
                description = String(description.dropFirst(3))
                parameters.append((name: firstName, type: typeName, description: description))
                index += 1
            }
            let callableName = context.makeUniqueName(name.text)
            let callableString = """
            @dynamicCallable struct \(callableName.text): DynamicCallable {
                @discardableResult
                func dynamicallyCall(withKeywordArguments args: [String: Any]) async throws -> String {
                    \(parameters.map {
                        "var \($0.name): \($0.type)!"
                    }.joined(separator: "\n"))
                    for (key, value) in args {
                        \(parameters.map {
                            "if key == \"\($0.name)\" { \($0.name) = value as! \($0.type) }"
                        }.joined(separator: "\n"))
                    }
            
                    let returnValue = try await \(name.text)(\(parameters.map { "\($0.name): \($0.name)" }.joined(separator: ",")))
                    let jsonValue = try JSONEncoder().encode(returnValue)
                    return String(data: jsonValue, encoding: .utf8)!
                }
            }
            """
            tools.append((name: name.text, description: description,
                          parameters: parameters,
                          callableString: callableString,
                          callableName: callableName.text))
        }
        
        
        return [
            .init(extendedType: type,
                inheritanceClause: .init(inheritedTypes: InheritedTypeListSyntax.init(arrayLiteral: .init(type: IdentifierTypeSyntax(name: "LlamaActor")))),
                  memberBlock: """
            {
                \(raw: tools.map {
                    $0.callableString
                }.joined(separator: "\n"))
            
                static var tools: [String: (DynamicCallable, _JSONFunctionSchema)] {
                    [\(raw: tools.map { tool in
                        """
                        "\(tool.name)": (\(tool.callableName)(), _JSONFunctionSchema(name: "\(tool.name)", description: "\(tool.description)", parameters: _JSONFunctionSchema.Parameters(properties: \(tool.parameters.count == 0 ? "[:]" : "[" + tool.parameters.map { parameter in
                            """
                            "\(parameter.name)": _JSONFunctionSchema.Property(type: \(parameter.type).self, description: "\(parameter.description)"),
                            """
                            }.joined() + "]"), required: [])))
                        """
                    }.joined(separator: ","))]
                }
            }
            """)
        ]
    }
}

@main
struct JSONSchemaMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        JSONSchemaMacro.self, LlamaActorMacro.self
    ]
}
