#!/usr/bin/env python3
import argparse
import json
import re
import sys

# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "?'

PRIMITIVE_RULES = {
    'boolean': '("true" | "false") space',
    'number': '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    'integer': '("-"? ([0-9] | [1-9] [0-9]*)) space',
    'string': r''' "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space ''',
    'null': '"null" space',
}

INVALID_RULE_CHARS_RE = re.compile(r'[^a-zA-Z0-9-]+')
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES = {'\r': '\\r', '\n': '\\n', '"': '\\"'}


class SchemaConverter:
    def __init__(self, prop_order):
        self._prop_order = prop_order
        self._rules = {'space': SPACE_RULE}

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), json.dumps(literal)
        )
        return f'"{escaped}"'

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub('-', name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f'{esc_name}{i}' in self._rules:
                i += 1
            key = f'{esc_name}{i}'
        self._rules[key] = rule
        return key

    def visit(self, schema, name):
        schema_type = schema.get('type')
        rule_name = name or 'root'

        if 'oneOf' in schema or 'anyOf' in schema:
            rule = ' | '.join((
                self.visit(alt_schema, f'{name}{"-" if name else ""}{i}')
                for i, alt_schema in enumerate(schema.get('oneOf') or schema['anyOf'])
            ))
            return self._add_rule(rule_name, rule)

        elif 'const' in schema:
            return self._add_rule(rule_name, self._format_literal(schema['const']))

        elif 'enum' in schema:
            rule = ' | '.join((self._format_literal(v) for v in schema['enum']))
            return self._add_rule(rule_name, rule)

        elif schema_type == 'object' and 'properties' in schema:
            # TODO: `required` keyword
            prop_order = self._prop_order
            prop_pairs = sorted(
                schema['properties'].items(),
                # sort by position in prop_order (if specified) then by key
                key=lambda kv: (prop_order.get(kv[0], len(prop_order)), kv[0]),
            )

            rule = '"{" space'
            for i, (prop_name, prop_schema) in enumerate(prop_pairs):
                prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
                if i > 0:
                    rule += ' "," space'
                rule += fr' {self._format_literal(prop_name)} space ":" space {prop_rule_name}'
            rule += ' "}" space'

            return self._add_rule(rule_name, rule)

        elif schema_type == 'array' and 'items' in schema:
            # TODO `prefixItems` keyword
            item_rule_name = self.visit(schema['items'], f'{name}{"-" if name else ""}item')
            list_item_operator = f'("," space {item_rule_name})'
            successive_items = ""
            min_items = schema.get("minItems", 0)
            if min_items > 0:
               first_item = f"({item_rule_name})"
               successive_items = list_item_operator * (min_items - 1)
               min_items -= 1
            else:
               first_item = f"({item_rule_name})?"
            max_items = schema.get("maxItems")
            if max_items is not None and max_items > min_items:
                successive_items += (list_item_operator + "?") * (max_items - min_items - 1)
            else:
                successive_items += list_item_operator + "*"
            rule = f'"[" space {first_item} {successive_items} "]" space'
            return self._add_rule(rule_name, rule)

        else:
            assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
            return self._add_rule(
                'root' if rule_name == 'root' else schema_type,
                PRIMITIVE_RULES[schema_type]
            )

    def format_grammar(self):
        return '\n'.join((f'{name} ::= {rule}' for name, rule in self._rules.items()))


def main(args_in = None):
    parser = argparse.ArgumentParser(
        description='''
            Generates a grammar (suitable for use in ./main) that produces JSON conforming to a
            given JSON schema. Only a subset of JSON schema features are supported; more may be
            added in the future.
        ''',
    )
    parser.add_argument(
        '--prop-order',
        default=[],
        type=lambda s: s.split(','),
        help='''
            comma-separated property names defining the order of precedence for object properties;
            properties not specified here are given lower precedence than those that are, and are
            sorted alphabetically
        '''
    )
    parser.add_argument('schema', help='file containing JSON schema ("-" for stdin)')
    args = parser.parse_args(args_in)

    schema = json.load(sys.stdin if args.schema == '-' else open(args.schema))
    prop_order = {name: idx for idx, name in enumerate(args.prop_order)}
    converter = SchemaConverter(prop_order)
    converter.visit(schema, '')
    print(converter.format_grammar())


if __name__ == '__main__':
    main()
