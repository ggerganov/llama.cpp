const SPACE_RULE = '" "*';

const PRIMITIVE_RULES = {
  boolean: '("true" | "false") space',
  number: '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
  integer: '("-"? ([0-9] | [1-9] [0-9]*)) space',
  value: 'object | array | string | number | boolean',
  object: '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
  array: '"[" space ( value ("," space value)* )? "]" space',
  string: ` "\\"" (
        [^"\\\\] |
        "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\\"" space`,
  null: '"null" space',
};

const INVALID_RULE_CHARS_RE = /[^\dA-Za-z-]+/g;
const GRAMMAR_LITERAL_ESCAPE_RE = /[\n\r]/g;
const GRAMMAR_LITERAL_ESCAPES = { '\r': '\\r', '\n': '\\n' };

export class SchemaConverter {
  constructor(propOrder) {
    this._propOrder = propOrder || {};
    this._rules = new Map();
    this._rules.set('space', SPACE_RULE);
    this.refBase = null;
  }

  _formatLiteral(literal) {
    const escaped = JSON.stringify(literal).slice(1, -1).replace(
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

  _resolveRef(ref) {
    // TODO: use https://github.com/APIDevTools/json-schema-ref-parser
    try {
      if (ref != null && ref.startsWith('#/')) {
        let target = this.refBase;
        let name = null;
        for (const sel of ref.split('/').slice(1)) {
          name = sel;
          target = target[sel];
        }
        return [name, target];
      }
      return null;
    } catch (e) {
      throw new Error(`Error resolving ref ${ref}: ${e}`);
    }
  }

  _generateUnionRule(name, altSchemas) {
    return altSchemas.map((altSchema, i) =>
      this.visit(altSchema, `${name}${name ? "-" : ""}${i}`)
    ).join(' | ');
  }

  _formatRangeChar(c) {
    if (c === '-' || c === ']' || c === '\\') {
      return '\\' + c;
    } else if (c === '\n') {
      return '\\n';
    } else if (c === '\r') {
      return '\\r';
    } else if (c === '\t') {
      return '\\t';
    } else {
      return c;
    }
  }

  _visitPattern(pattern) {
    if (!pattern.startsWith('^') || !pattern.endsWith('$')) {
      throw new Error('Pattern must start with "^" and end with "$"');
    }
    pattern = pattern.slice(1, -1);

    try {
      const visitSeq = seq => {
        const out = [];
        for (const [t, g] of groupBy(seq, x => x[0])) {
          const gList = Array.from(g);
          // Merge consecutive literals
          if (t === RegExp.LITERAL && gList.length > 1) {
            out.push(this._formatLiteral(gList.map(x => String.fromCharCode(x[1])).join('')));
          } else {
            out.push(...gList.map(visit));
          }
        }
        if (out.length === 1) {
          return out[0];
        }
        return '(' + out.join(' ') + ')';
      };

      const visit = pattern => {
        if (pattern[0] === RegExp.LITERAL) {
          return JSON.stringify(String.fromCharCode(pattern[1]));
        } else if (pattern[0] === RegExp.NOT_LITERAL) {
          return `[^${this._formatRangeChar(String.fromCharCode(pattern[1]))}]`;
        } else if (pattern[0] === RegExp.ANY) {
          throw new Error('Unsupported pattern: "."');
        } else if (pattern[0] === RegExp.IN) {
          const formatRangeComp = c => {
            if (c[0] === RegExp.LITERAL) {
              return this._formatRangeChar(String.fromCharCode(c[1]));
            } else if (c[0] === RegExp.RANGE) {
              return `${this._formatRangeChar(String.fromCharCode(c[1][0]))}-${this._formatRangeChar(String.fromCharCode(c[1][1]))}`;
            } else {
              throw new Error(`Unrecognized pattern: ${JSON.stringify(c)}`);
            }
          };
          return `[${pattern[1].map(formatRangeComp).join('')}]`;
        } else if (pattern[0] === RegExp.BRANCH) {
          return '(' + pattern[1][1].map(visit).join(' | ') + ')';
        } else if (pattern[0] === RegExp.SUBPATTERN) {
          return '(' + visit(pattern[1][3]) + ')';
        } else if (pattern[0] === RegExp.MAX_REPEAT) {
          const [minTimes, maxTimes, sub] = pattern[1];
          const subRule = visit(sub);

          if (minTimes === 0 && maxTimes == null) {
            return `${subRule}*`;
          } else if (minTimes === 0 && maxTimes === 1) {
            return `${subRule}?`;
          } else if (minTimes === 1 && maxTimes == null) {
            return `${subRule}+`;
          } else {
            return Array(minTimes).fill(subRule).concat(
              maxTimes != null ? Array(maxTimes - minTimes).fill(`${subRule}?`) : [`${subRule}*`]
            ).join(' ');
          }
        } else if (pattern instanceof RegExp.SubPattern) {
          return visitSeq(pattern.data);
        } else if (Array.isArray(pattern)) {
          return visitSeq(pattern);
        } else {
          throw new Error(`Unrecognized pattern: ${JSON.stringify(pattern)} (${typeof pattern})`);
        }
      };

      return visit(RegExp.parse(pattern));
    } catch (e) {
      throw new Error(`Error processing pattern: ${pattern}: ${e}`);
    }
  }

  visit(schema, name) {
    const oldRefBase = this.refBase;
    if ('definitions' in schema) {
      this.refBase = schema;
    }
    try {
      return this._visit(schema, name);
    } finally {
      this.refBase = oldRefBase;
    }
  }

  _visit(schema, name) {
    const schemaType = schema.type;
    const ref = schema.$ref;
    const ruleName = name || 'root';

    if ('oneOf' in schema || 'anyOf' in schema) {
      return this._addRule(ruleName, this._generateUnionRule(name, schema.oneOf || schema.anyOf));
    } else if (Array.isArray(schemaType)) {
      return this._addRule(ruleName, this._generateUnionRule(name, schemaType.map(t => ({ type: t }))));
    } else if ('const' in schema) {
      return this._addRule(ruleName, this._formatLiteral(schema.const));
    } else if ('enum' in schema) {
      const rule = schema.enum.map(v => this._formatLiteral(v)).join(' | ');
      return this._addRule(ruleName, rule);
    } else if ((schemaType == null || schemaType === 'object') && 'properties' in schema) {
      const required = new Set(schema.required || []);
      const { properties } = schema;
      return this._addRule(ruleName, this._buildObjectRule(Object.entries(properties), required, name));
    } else if (schemaType === 'object' && 'allOf' in schema) {
      const required = new Set();
      const properties = [];
      const addComponent = (compSchema, isRequired) => {
        const compRef = compSchema.$ref;
        if (compRef != null) {
          const resolved = this._resolveRef(compRef);
          if (resolved != null) {
            compSchema = resolved[1];
          }
        }

        if ('properties' in compSchema) {
          for (const [propName, propSchema] of Object.entries(compSchema.properties)) {
            properties.push([propName, propSchema]);
            if (isRequired) {
              required.add(propName);
            }
          }
        }
      };

      for (const t of schema.allOf) {
        if ('anyOf' in t) {
          for (const tt of t.anyOf) {
            addComponent(tt, false);
          }
        } else {
          addComponent(t, true);
        }
      }

      return this._addRule(ruleName, this._buildObjectRule(properties, required, name));
    } else if (schemaType === 'object' && 'additionalProperties' in schema) {
      let additionalProperties = schema.additionalProperties;
      if (typeof additionalProperties !== 'object') {
        additionalProperties = {};
      }

      const subName = `${name}${name ? "-" : ""}additionalProperties`;
      const valueRule = this.visit(additionalProperties, `${subName}-value`);
      const kvRule = this._addRule(`${subName}-kv`, `string ":" space ${valueRule}`);
      return this._addRule(
        ruleName,
        `( ${kvRule} ( "," space ${kvRule} )* )*`
      );
    } else if (schemaType === 'array' && 'items' in schema) {
      // TODO `prefixItems` keyword
      const { items } = schema;
      if (Array.isArray(items)) {
        return this._addRule(
          ruleName,
          '"[" space ' +
          items.map((item, i) => this.visit(item, `${name}-${i}`)).join(' "," space ') +
          ' "]" space'
        );
      } else {
        const itemRuleName = this.visit(items, `${name}${name ? "-" : ""}item`);
        const listItemOperator = `( "," space ${itemRuleName} )`;
        let successiveItems = "";
        const minItems = schema.minItems || 0;
        const maxItems = schema.maxItems;
        if (minItems > 0) {
          successiveItems = listItemOperator.repeat(minItems - 1);
        }
        if (maxItems != null && maxItems > minItems) {
          successiveItems += `${listItemOperator}?`.repeat(maxItems - minItems);
        } else {
          successiveItems += `${listItemOperator}*`;
        }
        const rule = minItems === 0
          ? `"[" space ( ${itemRuleName} ${successiveItems} )? "]" space`
          : `"[" space ${itemRuleName} ${successiveItems} "]" space`;
        return this._addRule(ruleName, rule);
      }
    } else if ((schemaType == null || schemaType === 'string') && 'pattern' in schema) {
      return this._addRule(ruleName, this._visitPattern(schema.pattern));
    } else if ((resolved = this._resolveRef(ref)) != null) {
      const [refName, definition] = resolved;
      const defName = name ? `${name}-${refName}` : '';
      return this.visit(definition, defName);
    // } else if (ref != null && ref.startsWith('https://')) {
    //   const refSchema = await fetch(ref).then(res => res.json());
    //   return this.visit(refSchema, ref);
    } else if ((schemaType === 'object' && Object.keys(schema).length === 1) || (schemaType == null && Object.keys(schema).length === 0)) {
      for (const [t, r] of Object.entries(PRIMITIVE_RULES)) {
        this._addRule(t, r);
      }
      return 'object';
    } else {
      if (!(schemaType in PRIMITIVE_RULES)) {
        throw new Error(`Unrecognized schema: ${JSON.stringify(schema)}`);
      }
      return this._addRule(
        ruleName === 'root' ? 'root' : schemaType,
        PRIMITIVE_RULES[schemaType]
      );
    }
  }

  _buildObjectRule(properties, required, name) {
    // TODO: `required` keyword
    const propOrder = this._propOrder;
    console.warn(`# properties: ${JSON.stringify(properties)}`);
    // sort by position in prop_order (if specified) then by original order
    const sortedProps = properties.map(([name]) => name).sort(
      (a, b) => (propOrder[a] ?? Infinity) - (propOrder[b] ?? Infinity)
    );

    const propKvRuleNames = {};
    for (const [propName, propSchema] of properties) {
      const propRuleName = this.visit(propSchema, `${name}${name ? "-" : ""}${propName}`);
      propKvRuleNames[propName] = this._addRule(
        `${name}${name ? "-" : ""}${propName}-kv`,
        `${this._formatLiteral(propName)} space ":" space ${propRuleName}`
      );
    }

    const requiredProps = sortedProps.filter(k => required.has(k));
    const optionalProps = sortedProps.filter(k => !required.has(k));

    let rule = '"{" space ';
    rule += requiredProps.map(k => propKvRuleNames[k]).join(' "," space ');

    if (optionalProps.length > 0) {
      rule += ' (';
      if (requiredProps.length > 0) {
        rule += ' "," space ( ';
      }

      const getRecursiveRefs = (ks, firstIsOptional) => {
        const [k, ...rest] = ks;
        const kvRuleName = propKvRuleNames[k];
        let res = firstIsOptional ? `( "," space ${kvRuleName} )?` : kvRuleName;
        if (rest.length > 0) {
          res += ' ' + this._addRule(
            `${name}${name ? "-" : ""}${k}-rest`,
            getRecursiveRefs(rest, true)
          );
        }
        return res;
      };

      rule += Array.from({ length: optionalProps.length }, (_, i) => getRecursiveRefs(optionalProps.slice(i), false)).join(' | ') + ' ';
      if (requiredProps.length > 0) {
        rule += ' ) ';
      }
      rule += ' )? ';
    }
    rule += ' "}" space ';

    return rule;
  }

  formatGrammar() {
    let grammar = '';
    this._rules.forEach((rule, name) => {
      grammar += `${name} ::= ${rule}\n`;
    });
    return grammar;
  }
}

// Helper function to group elements by a key function
function* groupBy(iterable, keyFn) {
  let lastKey = null;
  let group = [];
  for (const element of iterable) {
    const key = keyFn(element);
    if (lastKey !== null && key !== lastKey) {
      yield [lastKey, group];
      group = [];
    }
    group.push(element);
    lastKey = key;
  }
  if (group.length > 0) {
    yield [lastKey, group];
  }
}