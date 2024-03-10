#!/usr/bin/env python3
import argparse
import itertools
import json
import re
import sys
from typing import Any, Dict, List, Set, Tuple

# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "*'

PRIMITIVE_RULES = {
    'boolean': '("true" | "false") space',
    'number': '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    'integer': '("-"? ([0-9] | [1-9] [0-9]*)) space',
    'value'  : 'object | array | string | number | boolean',
    'object' : '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
    'array'  : '"[" space ( value ("," space value)* )? "]" space',
    'string': r''' "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space ''',
    'null': '"null" space',
}

INVALID_RULE_CHARS_RE = re.compile(r'[^a-zA-Z0-9-]+')
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n]')
GRAMMAR_LITERAL_ESCAPES = {'\r': '\\r', '\n': '\\n'}


class SchemaConverter:
    def __init__(self, prop_order):
        self._prop_order = prop_order
        self._rules = {'space': SPACE_RULE}
        self.ref_base = None

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), json.dumps(literal)[1:-1]
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


    def _format_range_char(self, c):
        if c in ('-', ']', '\\'):
            return '\\' + chr(c)
        elif c == '\n':
            return '\\n'
        elif c == '\r':
            return '\\r'
        elif c == '\t':
            return '\\t'
        else:
            return c

    def _visit_pattern(self, pattern, name):
        assert pattern.startswith('^') and pattern.endswith('$'), 'Pattern must start with "^" and end with "$"'
        pattern = pattern[1:-1]
        next_id = 1
        try:
            def visit_seq(seq):
                out = []
                for t, g in itertools.groupby(seq, lambda x: x[0]):
                    g = list(g)
                    # Merge consecutive literals
                    if t == re._parser.LITERAL and len(g) > 1:
                        out.append(self._format_literal(''.join(chr(x[1]) for x in g)))
                    else:
                        out.extend(visit(x) for x in g)
                if len(out) == 1:
                    return out[0]
                return '(' + ' '.join(out) + ')'
            
            def visit(pattern):
                nonlocal next_id

                if pattern[0] == re._parser.LITERAL:
                    return json.dumps(chr(pattern[1]))
                
                elif pattern[0] == re._parser.NOT_LITERAL:
                    return f'[^{self._format_range_char(chr(pattern[1]))}]'
                
                elif pattern[0] == re._parser.ANY:
                    raise ValueError('Unsupported pattern: "."')
                
                elif pattern[0] == re._parser.IN:
                    def format_range_comp(c):
                        if c[0] == re._parser.LITERAL:
                            return self._format_range_char(chr(c[1]))
                        elif c[0] == re._parser.RANGE:
                            return f'{self._format_range_char(chr(c[1][0]))}-{self._format_range_char(chr(c[1][1]))}'
                        else:
                            raise ValueError(f'Unrecognized pattern: {c}')
                    return f'[{"".join(format_range_comp(c) for c in pattern[1])}]'
                
                elif pattern[0] == re._parser.BRANCH:
                    return '(' + ' | '.join((visit(p) for p in pattern[1][1])) + ')'
                
                elif pattern[0] == re._parser.SUBPATTERN:
                    return '(' + visit(pattern[1][3]) + ')'
                
                elif pattern[0] == re._parser.MAX_REPEAT:
                    min_times = pattern[1][0]
                    max_times = pattern[1][1] if not pattern[1][1] == re._parser.MAXREPEAT else None
                    sub = visit(pattern[1][2])
                    sub = self._add_rule(f'{name}-{next_id}', sub)
                    next_id += 1

                    if min_times == 0 and max_times is None:
                        return f'{sub}*'
                    elif min_times == 0 and max_times == 1:
                        return f'{sub}?'
                    elif min_times == 1 and max_times is None:
                        return f'{sub}+'
                    else:
                        return ' '.join([sub] * min_times + 
                                        ([f'{sub}?'] * (max_times - min_times) if max_times is not None else [f'{sub}*']))
                
                elif isinstance(pattern, re._parser.SubPattern):
                    return visit_seq(pattern.data)
                
                elif isinstance(pattern, list):
                    return visit_seq(pattern)
                
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

        elif schema_type in (None, 'object') and 'properties' in schema:
            required = set(schema.get('required', []))
            properties = schema['properties']
            return self._add_rule(rule_name, self._build_object_rule(properties.items(), required, name))

        elif schema_type == 'object' and 'allOf' in schema:
            required = set()
            properties = []
            def add_component(comp_schema, is_required):
                ref = comp_schema.get('$ref')
                if ref is not None and (resolved := self._resolve_ref(ref)) is not None:
                    comp_schema = resolved[1]

                if 'properties' in comp_schema:
                    for prop_name, prop_schema in comp_schema['properties'].items():
                        properties.append((prop_name, prop_schema))
                        if is_required:
                            required.add(prop_name)

            for t in schema['allOf']:
                if 'anyOf' in t:
                    for tt in t['anyOf']:
                        add_component(tt, is_required=False)
                else:
                    add_component(t, is_required=True)

            return self._add_rule(rule_name, self._build_object_rule(properties, required, name))

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
                list_item_operator = f'( "," space {item_rule_name} )'
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
                    rule = f'"[" space ( {item_rule_name} {successive_items} )? "]" space'
                else:
                    rule = f'"[" space {item_rule_name} {successive_items} "]" space'
                return self._add_rule(rule_name, rule)
            
        elif schema_type in (None, 'string') and 'pattern' in schema:
            return self._add_rule(rule_name, self._visit_pattern(schema['pattern'], rule_name))

        elif (resolved := self._resolve_ref(ref)) is not None:
            (ref_name, definition) = resolved
            def_name = f'{name}-{ref_name}' if name else ''
            return self.visit(definition, def_name)
        
        elif ref is not None and ref.startswith('https://'):
            import requests
            ref_schema = requests.get(ref).json()
            return self.visit(ref_schema, ref)

        elif schema_type == 'object' and len(schema) == 1 or schema_type is None and len(schema) == 0:
            # return 'object'
            for t, r in PRIMITIVE_RULES.items():
                self._add_rule(t, r)
            return 'object'
        
        else:
            assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
            return self._add_rule(
                'root' if rule_name == 'root' else schema_type,
                PRIMITIVE_RULES[schema_type]
            )
    
    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], name: str):
        # TODO: `required` keyword
        prop_order = self._prop_order
        print(f'# properties: {properties}', file=sys.stderr)
        # sort by position in prop_order (if specified) then by original order
        sorted_props = [kv[0] for _, kv in sorted(enumerate(properties), key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]))]

        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                fr'{self._format_literal(prop_name)} space ":" space {prop_rule_name}'
            )

        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]
        
        rule = '"{" space '
        rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)            

        if optional_props:
            rule += ' ('
            if required_props:
                rule += ' "," space ( '

            def get_recursive_refs(ks, first_is_optional):
                [k, *rest] = ks
                kv_rule_name = prop_kv_rule_names[k]
                if first_is_optional:
                    res = f'( "," space {kv_rule_name} )?'
                else:
                    res = kv_rule_name
                if len(rest) > 0:
                    res += ' ' + self._add_rule(
                        f'{name}{"-" if name else ""}{k}-rest',
                        get_recursive_refs(rest, first_is_optional=True)
                    )
                return res

            rule += ' | '.join(
                get_recursive_refs(optional_props[i:], first_is_optional=False)
                for i in range(len(optional_props))
            ) + ' '
            if required_props:
                rule += ' ) '
            rule += ' )? '

        rule += ' "}" space '

        return rule

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
