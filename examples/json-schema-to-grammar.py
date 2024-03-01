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
        self.ref_base = None

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

    def _resolve_ref(self, ref):
        # TODO: use https://github.com/APIDevTools/json-schema-ref-parser
        try:
            if ref is not None and ref.startswith('#/'):# and 'definitions' in schema:
                target = self.ref_base
                name = None
                for sel in ref.split('/')[1:]:
                    name = sel
                    target = target[sel]
                return (name, target)
            return None
        except KeyError as e:
            raise Exception(f'Error resolving ref {ref}: {e}') from e
    
    def _generate_union_rule(self, name, alt_schemas):
        return ' | '.join((
            self.visit(alt_schema, f'{name}{"-" if name else ""}{i}')
            for i, alt_schema in enumerate(alt_schemas)
        ))

    def _visit_pattern(self, pattern):
        assert pattern.startswith('^') and pattern.endswith('$'), 'Pattern must start with "^" and end with "$"'
        pattern = pattern[1:-1]
        try:
            def visit(pattern):
                if pattern[0] == re._parser.LITERAL:
                    return json.dumps(chr(pattern[1]))
                elif pattern[0] == re._parser.ANY:
                    raise ValueError('Unsupported pattern: "."')
                elif pattern[0] == re._parser.IN:
                    def format_range_char(c):
                        if chr(c) in ('-', ']', '\\', '\n', '\r', '\t'):
                            return '\\' + chr(c)
                        else:
                            return chr(c)
                    def format_range_comp(c):
                        if c[0] == re._parser.LITERAL:
                            return format_range_char(c[1])
                        elif c[0] == re._parser.RANGE:
                            return f'{format_range_char(c[1][0])}-{format_range_char(c[1][1])}'
                        else:
                            raise ValueError(f'Unrecognized pattern: {c}')
                    return f'[{"".join(format_range_comp(c) for c in pattern[1])}]'
                elif pattern[0] == re._parser.BRANCH:
                    return ' | '.join((visit(p) for p in pattern[1][1]))
                elif pattern[0] == re._parser.SUBPATTERN:
                    return visit(pattern[1][3])
                elif pattern[0] == re._parser.MAX_REPEAT:
                    min_times = pattern[1][0]
                    max_times = pattern[1][1] if not pattern[1][1] == re._parser.MAXREPEAT else None
                    sub_pattern = pattern[1][2]
                    if min_times == 0 and max_times is None:
                        return f'{visit(sub_pattern)}*'
                    elif min_times == 0 and max_times == 1:
                        return f'{visit(sub_pattern)}?'
                    elif min_times == 1 and max_times is None:
                        return f'{visit(sub_pattern)}+'
                    else:
                        raise ValueError(f'Unrecognized pattern: {pattern} ({type(pattern)}; min: {min_times}, max: {max_times})')
                elif isinstance(pattern, re._parser.SubPattern):
                    return ' '.join(visit(p) for p in pattern.data)
                elif isinstance(pattern, list):# and (len(pattern) == 0 or isinstance(pattern[0], (tuple, list))):
                    return ' '.join(visit(p) for p in pattern)
                else:
                    raise ValueError(f'Unrecognized pattern: {pattern} ({type(pattern)})')
            return visit(re._parser.parse(pattern))
        except BaseException as e:
            raise Exception(f'Error processing pattern: {pattern}: {e}') from e

    def visit(self, schema, name):
        old_ref_base = self.ref_base
        if 'definitions' in schema:
            self.ref_base = schema
        try:
            return self._visit(schema, name)
        finally:
            self.ref_base = old_ref_base
    
    def _visit(self, schema, name):
        schema_type = schema.get('type')
        ref = schema.get('$ref')
        rule_name = name or 'root'

        if 'oneOf' in schema or 'anyOf' in schema:
            return self._add_rule(rule_name, self._generate_union_rule(name, schema.get('oneOf') or schema['anyOf']))
        
        elif isinstance(schema_type, list):
            return self._add_rule(rule_name, self._generate_union_rule(name, [{'type': t} for t in schema_type]))

        elif 'const' in schema:
            return self._add_rule(rule_name, self._format_literal(schema['const']))

        elif 'enum' in schema:
            rule = ' | '.join((self._format_literal(v) for v in schema['enum']))
            return self._add_rule(rule_name, rule)

        elif schema_type == 'object' and 'properties' in schema:
            # TODO: `required` keyword
            prop_order = self._prop_order
            required = set(schema.get('required', []))
            properties = schema['properties']
            # sort by position in prop_order (if specified) then by key
            def prop_sort_key(k):
                return (prop_order.get(k, len(prop_order)), k),

            prop_kv_rule_names = {}
            for prop_name, prop_schema in properties.items():
                prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
                prop_kv_rule_names[prop_name] = self._add_rule(
                    f'{name}{"-" if name else ""}{prop_name}-kv',
                    fr'{self._format_literal(prop_name)} space ":" space {prop_rule_name}'
                )

            req_props = list(sorted(
                (k for k in properties.keys() if k in required),
                key=prop_sort_key,
            ))
            opt_props = list(sorted(
                (k for k in properties.keys() if k not in required),
                key=prop_sort_key,
            ))

            def format_kv(prop_name, prop_rule_name):
                return fr'{self._format_literal(prop_name)} space ":" space {prop_rule_name}'
            
            rule = '"{" space '
            rule += ' "," space '.join(
                prop_kv_rule_names[prop_name]
                for prop_name in req_props
            )            
            if opt_props:
                rule += ' ('
                if req_props:
                    rule += ' "," space ( '

                def get_recursive_refs(ks, first_is_optional):
                    [prop_name, *rest] = ks
                    kv_rule_name = prop_kv_rule_names[prop_name]
                    if first_is_optional:
                        res = f'( "," space {kv_rule_name} )?'
                    else:
                        res = kv_rule_name
                    if len(rest) > 0:
                        res += ' ' + self._add_rule(
                            f'{name}{"-" if name else ""}{prop_name}-rest',
                            get_recursive_refs(rest, first_is_optional=True)
                        )
                    return res

                rule += ' | '.join(
                    get_recursive_refs(opt_props[i:], first_is_optional=False)
                    for i in range(len(opt_props))
                )
                if req_props:
                    rule += ')'
                rule += ')?'

            rule += '"}" space'

            return self._add_rule(rule_name, rule)

        elif schema_type == 'object' and 'additionalProperties' in schema:
            additional_properties = schema['additionalProperties']
            if not isinstance(additional_properties, dict):
                additional_properties = {}

            sub_name = f'{name}{"-" if name else ""}additionalProperties'
            value_rule = self.visit(additional_properties, f'{sub_name}-value')
            kv_rule = self._add_rule(f'{sub_name}-kv', f'string ":" space {value_rule}')
            return self._add_rule(
                rule_name,
                f'( {kv_rule} ( "," space {kv_rule} )* )*')

        elif schema_type == 'array' and 'items' in schema:
            # TODO `prefixItems` keyword
            items = schema['items']
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space ' +
                    ' "," space '.join(
                        self.visit(item, f'{name}-{i}')
                        for i, item in enumerate(items)) +
                    ' "]" space')
            else:
                item_rule_name = self.visit(items, f'{name}{"-" if name else ""}item')
                list_item_operator = f'("," space {item_rule_name})'
                successive_items = ""
                min_items = schema.get("minItems", 0)
                max_items = schema.get("maxItems")
                if min_items > 0:
                    successive_items = list_item_operator * (min_items - 1)
                    min_items -= 1
                if max_items is not None and max_items > min_items:
                    successive_items += (list_item_operator + "?") * (max_items - min_items - 1)
                else:
                    successive_items += list_item_operator + "*"
                if min_items == 0:
                    rule = f'"[" space ({item_rule_name} {successive_items})? "]" space'
                else:
                    rule = f'"[" space {item_rule_name} {successive_items} "]" space'
                return self._add_rule(rule_name, rule)
            
        elif schema_type in (None, 'string') and 'pattern' in schema:
            return self._add_rule(rule_name, self._visit_pattern(schema['pattern']))

        elif (resolved := self._resolve_ref(ref)) is not None:
            (ref_name, definition) = resolved
            def_name = f'{name}-{ref_name}' if name else ''
            return self.visit(definition, def_name)
        
        elif ref is not None and ref.startswith('https://'):
            import requests
            ref_schema = requests.get(ref).json()
            return self.visit(ref_schema, ref)

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

    if args.schema.startswith('https://'):
        import requests
        schema = requests.get(args.schema).json()
    elif args.schema == '-':
        schema = json.load(sys.stdin)
    else:
        with open(args.schema) as f:
            schema = json.load(f)
    prop_order = {name: idx for idx, name in enumerate(args.prop_order)}
    converter = SchemaConverter(prop_order)
    converter.visit(schema, '')
    print(converter.format_grammar())


if __name__ == '__main__':
    main()
