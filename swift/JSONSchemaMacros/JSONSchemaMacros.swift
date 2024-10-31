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

@main
struct JSONSchemaMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        JSONSchemaMacro.self
    ]
}
