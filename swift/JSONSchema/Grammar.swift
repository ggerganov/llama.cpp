import Foundation
import RegexBuilder

let SPACE_RULE = "\" \"?"

let PRIMITIVE_RULES: [String: String] = [
    "boolean": "(\"true\" | \"false\") space",
    "number": "\"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)? space",
    "integer": "\"-\"? ([0-9] | [1-9] [0-9]*) space",
    "string": "\"\\\"\" ([^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* \"\\\"\" space",
    "null": "\"null\" space",
]

let INVALID_RULE_CHARS_RE = try! NSRegularExpression(pattern: "[^a-zA-Z0-9-]+")
let GRAMMAR_LITERAL_ESCAPE_RE = try! NSRegularExpression(pattern: "[\r\n\"]")
let GRAMMAR_LITERAL_ESCAPES: [String: String] = ["\r": "\\r", "\n": "\\n", "\"": "\\\""]

public class SchemaConverter {
    private var propOrder: [String]
    private var rules: [String: String] = ["space": SPACE_RULE]

    public init(propOrder: [String]) {
        self.propOrder = propOrder
    }

    private func formatLiteral(_ literal: Any) -> String {
        let escaped = GRAMMAR_LITERAL_ESCAPES.reduce("\(literal)") {
            $0.replacingOccurrences(of: $1.key, with: $1.value)
        }
        
        return "\\\"\(escaped)\\\""
    }

    private func addRule(name: String, rule: String) -> String {
        let escName = INVALID_RULE_CHARS_RE.stringByReplacingMatches(
            in: name,
            options: [],
            range: NSRange(location: 0, length: name.count),
            withTemplate: "-"
        )

        var key = escName
        if let existingRule = rules[escName], existingRule != rule {
            var i = 0
            while rules["\(escName)\(i)"] != nil {
                i += 1
            }
            key = "\(escName)\(i)"
        }

        rules[key] = rule
        return key
    }

    public func visit(schema: [String: Any], name: String?) -> String {
        let schemaType = schema["type"] as? String
        let ruleName = name ?? "root"

        if let oneOf = schema["oneOf"] as? [[String: Any]] ?? schema["anyOf"] as? [[String: Any]] {
            let rule = oneOf.enumerated().map { (i, altSchema) in
                visit(schema: altSchema, name: "\(name ?? "")\(name != nil ? "-" : "")\(i)")
            }.joined(separator: " | ")
            return addRule(name: ruleName, rule: rule)
        } else if let constValue = schema["const"] {
            return addRule(name: ruleName, rule: formatLiteral(constValue))
        } else if let enumValues = schema["enum"] as? [Any] {
            let rule = enumValues.map { "\"\(formatLiteral($0))\"" }.joined(separator: " | ")
            return addRule(name: ruleName, rule: rule)
        } else if schemaType == "object", let properties = schema["properties"] as? [String: Any] {
            let propPairs = properties.sorted { (kv1, kv2) in
                let idx1 = propOrder.firstIndex(of: kv1.key) ?? propOrder.count
                let idx2 = propOrder.firstIndex(of: kv2.key) ?? propOrder.count
                return (idx1, kv1.key) < (idx2, kv2.key)
            }

            var rule = "\"{\" space"
            for (i, (propName, propSchema)) in propPairs.enumerated() {
                let propRuleName = visit(schema: propSchema as! [String : Any], name: "\(name ?? "")\(name != nil ? "-" : "")\(propName)")
                if i > 0 {
                    rule += " \",\" space"
                }
                rule += " \"\(formatLiteral(propName))\" space \":\" space \(propRuleName)"
            }
            rule += " \"}\" space"

            return addRule(name: ruleName, rule: rule)
        } else if schemaType == "array", let items = schema["items"] {
            let itemRuleName = visit(schema: items as! [String : Any], name: "\(name ?? "")\(name != nil ? "-" : "")item")
            let rule = "\"[\" space (\(itemRuleName) (\",\" space \(itemRuleName))*)? \"]\" space"
            return addRule(name: ruleName, rule: rule)
        } else {
            assert(PRIMITIVE_RULES.keys.contains(schemaType ?? ""), "Unrecognized schema: \(schema)")
            return addRule(name: ruleName == "root" ? "root" : schemaType!, rule: PRIMITIVE_RULES[schemaType!]!)
        }
    }

    public func formatGrammar() -> String {
        return rules.map { (name, rule) in "\(name) ::= \(rule)" }.joined(separator: "\n") + "\n"
    }
}
