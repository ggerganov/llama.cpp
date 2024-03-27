// WARNING: This file was ported from json-schema-to-grammar.py, please fix bugs / add features there first.
const SPACE_RULE = '" "?';

const PRIMITIVE_RULES = {
  boolean: '("true" | "false") space',
  number: '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
  integer: '("-"? ([0-9] | [1-9] [0-9]*)) space',
  value: 'object | array | string | number | boolean',
  object: '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
  array: '"[" space ( value ("," space value)* )? "]" space',
  uuid: '"\\"" ' + [8, 4, 4, 4, 12].map(n => [...new Array(n)].map(_ => '[0-9a-fA-F]').join('')).join(' "-" ') + ' "\\"" space',
  string: ` "\\"" (
        [^"\\\\] |
        "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\\"" space`,
  null: '"null" space',
};
const OBJECT_RULE_NAMES = ['object', 'array', 'string', 'number', 'boolean', 'null', 'value'];

// TODO: support "uri", "email" string formats
const DATE_RULES = {
    'date'   : '[0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( \"0\" [1-9] | [1-2] [0-9] | "3" [0-1] )',
    'time'   : '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
    'date-time': 'date "T" time',
    'date-string': '"\\"" date "\\"" space',
    'time-string': '"\\"" time "\\"" space',
    'date-time-string': '"\\"" date-time "\\"" space',
};

const RESERVED_NAMES = {'root': true, ...PRIMITIVE_RULES, ...DATE_RULES};

const INVALID_RULE_CHARS_RE = /[^\dA-Za-z-]+/g;
const GRAMMAR_LITERAL_ESCAPE_RE = /[\n\r"]/g;
const GRAMMAR_RANGE_LITERAL_ESCAPE_RE = /[\n\r"\]\-\\]/g;
const GRAMMAR_LITERAL_ESCAPES = { '\r': '\\r', '\n': '\\n', '"': '\\"', '-': '\\-', ']': '\\]' };

const NON_LITERAL_SET = new Set('|.()[]{}*+?');
const ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = new Set('[]()|{}*+?');

export class SchemaConverter {
  constructor(options) {
    this._propOrder = options.prop_order || {};
    this._allowFetch = options.allow_fetch || false;
    this._dotall = options.dotall || false;
    this._rules = {'space': SPACE_RULE};
    this._refs = {};
    this._refsBeingResolved = new Set();
  }

  _formatLiteral(literal) {
    const escaped = literal.replace(
      GRAMMAR_LITERAL_ESCAPE_RE,
      m => GRAMMAR_LITERAL_ESCAPES[m]
    );
    return `"${escaped}"`;
  }

  _formatRangeChar(literal) {
    return JSON.stringify(literal).slice(1, -1).replace(
      GRAMMAR_RANGE_LITERAL_ESCAPE_RE,
      m => GRAMMAR_LITERAL_ESCAPES[m]
    );
  }

  _addRule(name, rule) {
    let escName = name.replace(INVALID_RULE_CHARS_RE, '-');
    let key = escName;

    if (escName in this._rules) {
      if (this._rules[escName] === rule) {
        return key;
      }

      let i = 0;
      while ((`${escName}${i}` in this._rules) && (this._rules[`${escName}${i}`] !== rule)) {
        i += 1;
      }
      key = `${escName}${i}`;
    }

    this._rules[key] = rule;
    return key;
  }

  async resolveRefs(schema, url) {
    const visit = async (n) => {
      if (Array.isArray(n)) {
        return Promise.all(n.map(visit));
      } else if (typeof n === 'object' && n !== null) {
        let ref = n.$ref;
        let target;
        if (ref !== undefined && !this._refs[ref]) {
          if (ref.startsWith('https://')) {
            if (!this._allowFetch) {
              throw new Error('Fetching remote schemas is not allowed (use --allow-fetch for force)');
            }
            const fetch = (await import('node-fetch')).default;

            const fragSplit = ref.split('#');
            const baseUrl = fragSplit[0];

            target = this._refs[baseUrl];
            if (!target) {
              target = await this.resolveRefs(await fetch(ref).then(res => res.json()), baseUrl);
              this._refs[baseUrl] = target;
            }

            if (fragSplit.length === 1 || fragSplit[fragSplit.length - 1] === '') {
              return target;
            }
          } else if (ref.startsWith('#/')) {
            target = schema;
            ref = `${url}${ref}`;
            n.$ref = ref;
          } else {
            throw new Error(`Unsupported ref ${ref}`);
          }

          const selectors = ref.split('#')[1].split('/').slice(1);
          for (const sel of selectors) {
            if (!target || !(sel in target)) {
              throw new Error(`Error resolving ref ${ref}: ${sel} not in ${JSON.stringify(target)}`);
            }
            target = target[sel];
          }

          this._refs[ref] = target;
        } else {
          await Promise.all(Object.values(n).map(visit));
        }
      }

      return n;
    };

    return visit(schema);
  }

  _generateUnionRule(name, altSchemas) {
    return altSchemas
      .map((altSchema, i) => this.visit(altSchema, `${name ?? ''}${name ? '-' : 'alternative-'}${i}`))
      .join(' | ');
  }

  _visitPattern(pattern, name) {
    if (!pattern.startsWith('^') || !pattern.endsWith('$')) {
      throw new Error('Pattern must start with "^" and end with "$"');
    }
    pattern = pattern.slice(1, -1);
    const subRuleIds = {};

    let i = 0;
    const length = pattern.length;

    const getDot = () => {
      let rule;
      if (this._dotall) {
        rule = '[\\U00000000-\\U0010FFFF]';
      } else {
        // Accept any character... except \n and \r line break chars (\x0A and \xOD)
        rule = '[\\U00000000-\\x09\\x0B\\x0C\\x0E-\\U0010FFFF]';
      }
      return this._addRule('dot', rule);
    };


    const toRule = ([s, isLiteral]) => isLiteral ? "\"" + s + "\"" : s;

    const transform = () => {
      const start = i;
      // For each component of this sequence, store its string representation and whether it's a literal.
      // We only need a flat structure here to apply repetition operators to the last item, and
      // to merge literals at the and (we're parsing grouped ( sequences ) recursively and don't treat '|' specially
      // (GBNF's syntax is luckily very close to regular expressions!)
      const seq = [];

      const joinSeq = () => {
        const ret = [];
        for (const [isLiteral, g] of groupBy(seq, x => x[1])) {
          if (isLiteral) {
            ret.push([[...g].map(x => x[0]).join(''), true]);
          } else {
            ret.push(...g);
          }
        }
        if (ret.length === 1) {
          return ret[0];
        }
        return [ret.map(x => toRule(x)).join(' '), false];
      };

      while (i < length) {
        const c = pattern[i];
        if (c === '.') {
          seq.push([getDot(), false]);
          i += 1;
        } else if (c === '(') {
          i += 1;
          if (i < length) {
            if (pattern[i] === '?') {
              throw new Error(`Unsupported pattern syntax "${pattern[i]}" at index ${i} of /${pattern}/`);
            }
          }
          seq.push([`(${toRule(transform())})`, false]);
        } else if (c === ')') {
          i += 1;
          if (start <= 0 || pattern[start - 1] !== '(') {
            throw new Error(`Unbalanced parentheses; start = ${start}, i = ${i}, pattern = ${pattern}`);
          }
          return joinSeq();
        } else if (c === '[') {
          let squareBrackets = c;
          i += 1;
          while (i < length && pattern[i] !== ']') {
            if (pattern[i] === '\\') {
              squareBrackets += pattern.slice(i, i + 2);
              i += 2;
            } else {
              squareBrackets += pattern[i];
              i += 1;
            }
          }
          if (i >= length) {
            throw new Error(`Unbalanced square brackets; start = ${start}, i = ${i}, pattern = ${pattern}`);
          }
          squareBrackets += ']';
          i += 1;
          seq.push([squareBrackets, false]);
        } else if (c === '|') {
          seq.push(['|', false]);
          i += 1;
        } else if (c === '*' || c === '+' || c === '?') {
          seq[seq.length - 1] = [toRule(seq[seq.length - 1]) + c, false];
          i += 1;
        } else if (c === '{') {
          let curlyBrackets = c;
          i += 1;
          while (i < length && pattern[i] !== '}') {
            curlyBrackets += pattern[i];
            i += 1;
          }
          if (i >= length) {
            throw new Error(`Unbalanced curly brackets; start = ${start}, i = ${i}, pattern = ${pattern}`);
          }
          curlyBrackets += '}';
          i += 1;
          const nums = curlyBrackets.slice(1, -1).split(',').map(s => s.trim());
          let minTimes, maxTimes;
          if (nums.length === 1) {
            minTimes = parseInt(nums[0], 10);
            maxTimes = minTimes;
          } else {
            if (nums.length !== 2) {
              throw new Error(`Invalid quantifier ${curlyBrackets}`);
            }
            minTimes = nums[0] ? parseInt(nums[0], 10) : 0;
            maxTimes = nums[1] ? parseInt(nums[1], 10) : Infinity;
          }

          let [sub, subIsLiteral] = seq[seq.length - 1];

          if (minTimes === 0 && maxTimes === Infinity) {
            seq[seq.length - 1] = [`${sub}*`, false];
          } else if (minTimes === 0 && maxTimes === 1) {
            seq[seq.length - 1] = [`${sub}?`, false];
          } else if (minTimes === 1 && maxTimes === Infinity) {
            seq[seq.length - 1] = [`${sub}+`, false];
          } else {
            if (!subIsLiteral) {
              let id = subRuleIds[sub];
              if (id === undefined) {
                id = this._addRule(`${name}-${Object.keys(subRuleIds).length + 1}`, sub);
                subRuleIds[sub] = id;
              }
              sub = id;
            }

            const repeatedSub = Array.from({ length: minTimes }, () => subIsLiteral ? `"${sub.slice(1, -1).repeat(minTimes)}"` : sub);
            const optionalSub = maxTimes !== undefined ? Array.from({ length: maxTimes - minTimes }, () => `${sub}?`) : [`${sub}*`];
            seq[seq.length - 1] = [repeatedSub.concat(optionalSub).join(' '), false];
          }
        } else {
          let literal = '';
          while (i < length) {
            if (pattern[i] === '\\' && i < length - 1) {
              const next = pattern[i + 1];
              if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.has(next)) {
                i += 1;
                literal += pattern[i];
                i += 1;
              } else {
                literal += pattern.slice(i, i + 2);
                i += 2;
              }
            } else if (pattern[i] === '"') {
              literal += '\\"';
              i += 1;
            } else if (!NON_LITERAL_SET.has(pattern[i]) &&
                (i === length - 1 || literal === '' || pattern[i + 1] === '.' || !NON_LITERAL_SET.has(pattern[i+1]))) {
              literal += pattern[i];
              i += 1;
            } else {
              break;
            }
          }
          if (literal !== '') {
            seq.push([literal, true]);
          }
        }
      }

      return joinSeq();
    };

    return this._addRule(name, "\"\\\"\" " + toRule(transform()) + " \"\\\"\" space")
  }

  _resolveRef(ref) {
    let refName = ref.split('/').pop();
    if (!(refName in this._rules) && !this._refsBeingResolved.has(ref)) {
      this._refsBeingResolved.add(ref);
      const resolved = this._refs[ref];
      refName = this.visit(resolved, refName);
      this._refsBeingResolved.delete(ref);
    }
    return refName;
  }

  _generateConstantRule(value) {
    return this._formatLiteral(JSON.stringify(value));
  }

  visit(schema, name) {
    const schemaType = schema.type;
    const schemaFormat = schema.format;
    const ruleName = name in RESERVED_NAMES ? name + '-' : name == '' ? 'root' : name;

    const ref = schema.$ref;
    if (ref !== undefined) {
      return this._addRule(ruleName, this._resolveRef(ref));
    } else if (schema.oneOf || schema.anyOf) {
      return this._addRule(ruleName, this._generateUnionRule(name, schema.oneOf || schema.anyOf));
    } else if (Array.isArray(schemaType)) {
      return this._addRule(ruleName, this._generateUnionRule(name, schemaType.map(t => ({ type: t }))));
    } else if ('const' in schema) {
      return this._addRule(ruleName, this._generateConstantRule(schema.const));
    } else if ('enum' in schema) {
      const rule = schema.enum.map(v => this._generateConstantRule(v)).join(' | ');
      return this._addRule(ruleName, rule);
    } else if ((schemaType === undefined || schemaType === 'object') &&
               ('properties' in schema ||
                ('additionalProperties' in schema && schema.additionalProperties !== true))) {
      const required = new Set(schema.required || []);
      const properties = Object.entries(schema.properties ?? {});
      return this._addRule(ruleName, this._buildObjectRule(properties, required, name, schema.additionalProperties));
    } else if ((schemaType === undefined || schemaType === 'object') && 'allOf' in schema) {
      const required = new Set();
      const properties = [];
      const addComponent = (compSchema, isRequired) => {
        const ref = compSchema.$ref;
        if (ref !== undefined) {
          compSchema = this._refs[ref];
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

      return this._addRule(ruleName, this._buildObjectRule(properties, required, name, /* additionalProperties= */ false));
    } else if ((schemaType === undefined || schemaType === 'array') && ('items' in schema || 'prefixItems' in schema)) {
      const items = schema.items ?? schema.prefixItems;
      if (Array.isArray(items)) {
        return this._addRule(
          ruleName,
          '"[" space ' +
            items.map((item, i) => this.visit(item, `${name ?? ''}${name ? '-' : ''}tuple-${i}`)).join(' "," space ') +
            ' "]" space'
        );
      } else {
        const itemRuleName = this.visit(items, `${name ?? ''}${name ? '-' : ''}item`);
        const listItemOperator = `( "," space ${itemRuleName} )`;
        let successiveItems = '';
        let minItems = schema.minItems || 0;
        const maxItems = schema.maxItems;
        if (minItems > 0) {
          successiveItems = listItemOperator.repeat(minItems - 1);
          minItems--;
        }
        if (maxItems !== undefined && maxItems > minItems) {
          successiveItems += `${listItemOperator}?`.repeat(maxItems - minItems - 1);
        } else {
          successiveItems += `${listItemOperator}*`;
        }
        const rule = minItems === 0
          ? `"[" space ( ${itemRuleName} ${successiveItems} )? "]" space`
          : `"[" space ${itemRuleName} ${successiveItems} "]" space`;
        return this._addRule(ruleName, rule);
      }
    } else if ((schemaType === undefined || schemaType === 'string') && 'pattern' in schema) {
      return this._visitPattern(schema.pattern, ruleName);
    } else if ((schemaType === undefined || schemaType === 'string') && /^uuid[1-5]?$/.test(schema.format || '')) {
      return this._addRule(
          ruleName === 'root' ? 'root' : schemaFormat,
          PRIMITIVE_RULES['uuid'])
    } else if ((schemaType === undefined || schemaType === 'string') && schema.format in DATE_RULES) {
      for (const [t, r] of Object.entries(DATE_RULES)) {
        this._addRule(t, r);
      }
      return schemaFormat + '-string';
    } else if ((schemaType === 'object') || (Object.keys(schema).length === 0)) {
      for (const n of OBJECT_RULE_NAMES) {
        this._addRule(n, PRIMITIVE_RULES[n]);
      }
      return this._addRule(ruleName, 'object');
    } else {
      if (!(schemaType in PRIMITIVE_RULES)) {
        throw new Error(`Unrecognized schema: ${JSON.stringify(schema)}`);
      }
      // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
      return this._addRule(ruleName === 'root' ? 'root' : schemaType, PRIMITIVE_RULES[schemaType]);
    }
  }

  _buildObjectRule(properties, required, name, additionalProperties) {
    const propOrder = this._propOrder;
    // sort by position in prop_order (if specified) then by original order
    const sortedProps = properties.map(([k]) => k).sort((a, b) => {
      const orderA = propOrder[a] || Infinity;
      const orderB = propOrder[b] || Infinity;
      return orderA - orderB || properties.findIndex(([k]) => k === a) - properties.findIndex(([k]) => k === b);
    });

    const propKvRuleNames = {};
    for (const [propName, propSchema] of properties) {
      const propRuleName = this.visit(propSchema, `${name ?? ''}${name ? '-' : ''}${propName}`);
      propKvRuleNames[propName] = this._addRule(
        `${name ?? ''}${name ? '-' : ''}${propName}-kv`,
        `${this._formatLiteral(JSON.stringify(propName))} space ":" space ${propRuleName}`
      );
    }
    const requiredProps = sortedProps.filter(k => required.has(k));
    const optionalProps = sortedProps.filter(k => !required.has(k));

    if (typeof additionalProperties === 'object' || additionalProperties === true) {
      const subName = `${name ?? ''}${name ? '-' : ''}additional`;
      const valueRule = this.visit(additionalProperties === true ? {} : additionalProperties, `${subName}-value`);
      propKvRuleNames['*'] = this._addRule(
        `${subName}-kv`,
        `${this._addRule('string', PRIMITIVE_RULES['string'])} ":" space ${valueRule}`);
      optionalProps.push('*');
    }

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
        let res;
        if (k === '*') {
            res = this._addRule(
                `${name ?? ''}${name ? '-' : ''}additional-kvs`,
                `${kvRuleName} ( "," space ` + kvRuleName + ` )*`
            )
        } else if (firstIsOptional) {
          res = `( "," space ${kvRuleName} )?`;
        } else {
          res = kvRuleName;
        }
        if (rest.length > 0) {
          res += ' ' + this._addRule(
            `${name ?? ''}${name ? '-' : ''}${k}-rest`,
            getRecursiveRefs(rest, true)
          );
        }
        return res;
      };

      rule += optionalProps.map((_, i) => getRecursiveRefs(optionalProps.slice(i), false)).join(' | ');
      if (requiredProps.length > 0) {
        rule += ' )';
      }
      rule += ' )?';
    }

    rule += ' "}" space';

    return rule;
  }

  formatGrammar() {
    let grammar = '';
    for (const [name, rule] of Object.entries(this._rules).sort(([a], [b]) => a.localeCompare(b))) {
      grammar += `${name} ::= ${rule}\n`;
    }
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
