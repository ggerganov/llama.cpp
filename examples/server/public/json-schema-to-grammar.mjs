const SPACE_RULE = '" "?';

const PRIMITIVE_RULES = {
  boolean: '("true" | "false") space',
  number: '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
  integer: '("-"? ([0-9] | [1-9] [0-9]*)) space',
  string: ` "\\"" (
        [^"\\\\] |
        "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\\"" space`,
  null: '"null" space',
};

const INVALID_RULE_CHARS_RE = /[^\dA-Za-z-]+/g;
const GRAMMAR_LITERAL_ESCAPE_RE = /[\n\r"]/g;
const GRAMMAR_LITERAL_ESCAPES = {'\r': '\\r', '\n': '\\n', '"': '\\"'};

export class SchemaConverter {
  constructor(propOrder) {
    this._propOrder = propOrder || {};
    this._rules = new Map();
    this._rules.set('space', SPACE_RULE);
  }

  _formatLiteral(literal) {
    const escaped = JSON.stringify(literal).replace(
      GRAMMAR_LITERAL_ESCAPE_RE,
      m => GRAMMAR_LITERAL_ESCAPES[m]
    );
    return `"${escaped}"`;
  }

  _addRule(name, rule) {
    let escName = name.replace(INVALID_RULE_CHARS_RE, '-');
    let key = escName;

    if (this._rules.has(escName)) {
      if (this._rules.get(escName) === rule) {
        return key;
      }

      let i = 0;
      while (this._rules.has(`${escName}${i}`)) {
        i += 1;
      }
      key = `${escName}${i}`;
    }

    this._rules.set(key, rule);
    return key;
  }

  visit(schema, name) {
    const schemaType = schema.type;
    const ruleName = name || 'root';

    if (schema.oneOf || schema.anyOf) {
      const rule = (schema.oneOf || schema.anyOf).map((altSchema, i) =>
        this.visit(altSchema, `${name}${name ? "-" : ""}${i}`)
      ).join(' | ');

      return this._addRule(ruleName, rule);
    } else if ('const' in schema) {
      return this._addRule(ruleName, this._formatLiteral(schema.const));
    } else if ('enum' in schema) {
      const rule = schema.enum.map(v => this._formatLiteral(v)).join(' | ');
      return this._addRule(ruleName, rule);
    } else if (schemaType === 'object' && 'properties' in schema) {
      // TODO: `required` keyword (from python implementation)
      const propOrder = this._propOrder;
      const propPairs = Object.entries(schema.properties).sort((a, b) => {
        // sort by position in prop_order (if specified) then by key
        const orderA = typeof propOrder[a[0]] === 'number' ? propOrder[a[0]] : Infinity;
        const orderB = typeof propOrder[b[0]] === 'number' ? propOrder[b[0]] : Infinity;
        return orderA - orderB || a[0].localeCompare(b[0]);
      });

      let rule = '"{" space';
      propPairs.forEach(([propName, propSchema], i) => {
        const propRuleName = this.visit(propSchema, `${name}${name ? "-" : ""}${propName}`);
        if (i > 0) {
          rule += ' "," space';
        }
        rule += ` ${this._formatLiteral(propName)} space ":" space ${propRuleName}`;
      });
      rule += ' "}" space';

      return this._addRule(ruleName, rule);
    } else if (schemaType === 'array' && 'items' in schema) {
      // TODO `prefixItems` keyword (from python implementation)
      const itemRuleName = this.visit(schema.items, `${name}${name ? "-" : ""}item`);
      const rule = `"[" space (${itemRuleName} ("," space ${itemRuleName})*)? "]" space`;
      return this._addRule(ruleName, rule);
    } else {
      if (!PRIMITIVE_RULES[schemaType]) {
        throw new Error(`Unrecognized schema: ${JSON.stringify(schema)}`);
      }
      return this._addRule(
        ruleName === 'root' ? 'root' : schemaType,
        PRIMITIVE_RULES[schemaType]
      );
    }
  }

  formatGrammar() {
    let grammar = '';
    this._rules.forEach((rule, name) => {
      grammar += `${name} ::= ${rule}\n`;
    });
    return grammar;
  }
}
