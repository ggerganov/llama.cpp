#!/usr/bin/env python3
import argparse
import itertools
import json
import re
import sys
from typing import Any, Dict, List, Set, Tuple, Union

# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "?'

PRIMITIVE_RULES = {
    'boolean': '("true" | "false") space',
    'number': '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    'integer': '("-"? ([0-9] | [1-9] [0-9]*)) space',
    'value'  : 'object | array | string | number | boolean',
    'object' : '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
    'array'  : '"[" space ( value ("," space value)* )? "]" space',
    'uuid'   : '"\\"" ' + ' "-" '.join('[0-9a-fA-F]' * n for n in [8, 4, 4, 4, 12]) + ' "\\"" space',
    'string': r''' "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space''',
    'null': '"null" space',
}
OBJECT_RULE_NAMES = ['object', 'array', 'string', 'number', 'boolean', 'null', 'value']

# TODO: support "uri", "email" string formats
DATE_RULES = {
    'date'   : '[0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( \"0\" [1-9] | [1-2] [0-9] | "3" [0-1] )',
    'time'   : '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
    'date-time': 'date "T" time',
    'date-string': '"\\"" date "\\"" space',
    'time-string': '"\\"" time "\\"" space',
    'date-time-string': '"\\"" date-time "\\"" space',
}

RESERVED_NAMES = set(["root", *PRIMITIVE_RULES.keys(), *DATE_RULES.keys()])

INVALID_RULE_CHARS_RE = re.compile(r'[^a-zA-Z0-9-]+')
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_RANGE_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"\]\-\\]')
GRAMMAR_LITERAL_ESCAPES = {'\r': '\\r', '\n': '\\n', '"': '\\"', '-': '\\-', ']': '\\]'}

NON_LITERAL_SET = set('|.()[]{}*+?')
ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = set('[]()|{}*+?')

DATE_PATTERN = '[0-9]{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])'
TIME_PATTERN = '([01][0-9]|2[0-3])(:[0-5][0-9]){2}(\\.[0-9]{1,3})?(Z|[+-](([01][0-9]|2[0-3]):[0-5][0-9]))' # Cap millisecond precision w/ 3 digits

class SchemaConverter:
    def __init__(self, *, prop_order, allow_fetch, dotall, raw_pattern):
        self._prop_order = prop_order
        self._allow_fetch = allow_fetch
        self._dotall = dotall
        self._raw_pattern = raw_pattern
        self._rules = {'space': SPACE_RULE}
        self._refs = {}
        self._refs_being_resolved = set()

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), literal
        )
        return f'"{escaped}"'

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub('-', name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f'{esc_name}{i}' in self._rules and self._rules[f'{esc_name}{i}'] != rule:
                i += 1
            key = f'{esc_name}{i}'
        self._rules[key] = rule
        return key

    def resolve_refs(self, schema: dict, url: str):
        '''
            Resolves all $ref fields in the given schema, fetching any remote schemas,
            replacing $ref with absolute reference URL and populating self._refs with the
            respective referenced (sub)schema dictionaries.
        '''
        def visit(n: dict):
            if isinstance(n, list):
                return [visit(x) for x in n]
            elif isinstance(n, dict):
                ref = n.get('$ref')
                if ref is not None and ref not in self._refs:
                    if ref.startswith('https://'):
                        assert self._allow_fetch, 'Fetching remote schemas is not allowed (use --allow-fetch for force)'
                        import requests

                        frag_split = ref.split('#')
                        base_url = frag_split[0]

                        target = self._refs.get(base_url)
                        if target is None:
                            target = self.resolve_refs(requests.get(ref).json(), base_url)
                            self._refs[base_url] = target

                        if len(frag_split) == 1 or frag_split[-1] == '':
                            return target
                    elif ref.startswith('#/'):
                        target = schema
                        ref = f'{url}{ref}'
                        n['$ref'] = ref
                    else:
                        raise ValueError(f'Unsupported ref {ref}')

                    for sel in ref.split('#')[-1].split('/')[1:]:
                        assert target is not None and sel in target, f'Error resolving ref {ref}: {sel} not in {target}'
                        target = target[sel]

                    self._refs[ref] = target
                else:
                    for v in n.values():
                        visit(v)

            return n
        return visit(schema)

    def _generate_union_rule(self, name, alt_schemas):
        return ' | '.join((
            self.visit(alt_schema, f'{name}{"-" if name else "alternative-"}{i}')
            for i, alt_schema in enumerate(alt_schemas)
        ))

    def _visit_pattern(self, pattern, name):
        '''
            Transforms a regular expression pattern into a GBNF rule.

            Input: https://json-schema.org/understanding-json-schema/reference/regular_expressions
            Output: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

            Unsupported features: negative/positive lookaheads, greedy/non-greedy modifiers.

            Mostly a 1:1 translation, except for {x} / {x,} / {x,y} quantifiers for which
            we define sub-rules to keep the output lean.
        '''

        assert pattern.startswith('^') and pattern.endswith('$'), 'Pattern must start with "^" and end with "$"'
        pattern = pattern[1:-1]
        sub_rule_ids = {}

        i = 0
        length = len(pattern)

        def to_rule(s: Tuple[str, bool]) -> str:
            (txt, is_literal) = s
            return "\"" + txt + "\"" if is_literal else txt

        def transform() -> Tuple[str, bool]:
            '''
                Parse a unit at index i (advancing it), and return its string representation + whether it's a literal.
            '''
            nonlocal i
            nonlocal pattern
            nonlocal sub_rule_ids

            start = i
            # For each component of this sequence, store its string representation and whether it's a literal.
            # We only need a flat structure here to apply repetition operators to the last item, and
            # to merge literals at the and (we're parsing grouped ( sequences ) recursively and don't treat '|' specially
            # (GBNF's syntax is luckily very close to regular expressions!)
            seq: list[Tuple[str, bool]] = []

            def get_dot():
                if self._dotall:
                    rule = '[\\U00000000-\\U0010FFFF]'
                else:
                    # Accept any character... except \n and \r line break chars (\x0A and \xOD)
                    rule = '[\\U00000000-\\x09\\x0B\\x0C\\x0E-\\U0010FFFF]'
                return self._add_rule(f'dot', rule)

            def join_seq():
                nonlocal seq
                ret = []
                for is_literal, g in itertools.groupby(seq, lambda x: x[1]):
                    if is_literal:
                        ret.append((''.join(x[0] for x in g), True))
                    else:
                        ret.extend(g)
                if len(ret) == 1:
                    return ret[0]
                return (' '.join(to_rule(x) for x in seq), False)

            while i < length:
                c = pattern[i]
                if c == '.':
                    seq.append((get_dot(), False))
                    i += 1
                elif c == '(':
                    i += 1
                    if i < length:
                        assert pattern[i] != '?', f'Unsupported pattern syntax "{pattern[i]}" at index {i} of /{pattern}/'
                    seq.append((f'({to_rule(transform())})', False))
                elif c == ')':
                    i += 1
                    assert start > 0 and pattern[start-1] == '(', f'Unbalanced parentheses; start = {start}, i = {i}, pattern = {pattern}'
                    return join_seq()
                elif c == '[':
                    square_brackets = c
                    i += 1
                    while i < length and pattern[i] != ']':
                        if pattern[i] == '\\':
                            square_brackets += pattern[i:i+2]
                            i += 2
                        else:
                            square_brackets += pattern[i]
                            i += 1
                    assert i < length, f'Unbalanced square brackets; start = {start}, i = {i}, pattern = {pattern}'
                    square_brackets += ']'
                    i += 1
                    seq.append((square_brackets, False))
                elif c == '|':
                    seq.append(('|', False))
                    i += 1
                elif c in ('*', '+', '?'):
                    seq[-1] = (to_rule(seq[-1]) + c, False)
                    i += 1
                elif c == '{':
                    curly_brackets = c
                    i += 1
                    while i < length and pattern[i] != '}':
                        curly_brackets += pattern[i]
                        i += 1
                    assert i < length, f'Unbalanced curly brackets; start = {start}, i = {i}, pattern = {pattern}'
                    curly_brackets += '}'
                    i += 1
                    nums = [s.strip() for s in curly_brackets[1:-1].split(',')]
                    min_times = 0
                    max_times = None
                    try:
                        if len(nums) == 1:
                            min_times = int(nums[0])
                            max_times = min_times
                        else:
                            assert len(nums) == 2
                            min_times = int(nums[0]) if nums[0] else 0
                            max_times = int(nums[1]) if nums[1] else None
                    except ValueError:
                        raise ValueError(f'Invalid quantifier {curly_brackets} in /{pattern}/')

                    (sub, sub_is_literal) = seq[-1]

                    if min_times == 0 and max_times is None:
                        seq[-1] = (f'{sub}*', False)
                    elif min_times == 0 and max_times == 1:
                        seq[-1] = (f'{sub}?', False)
                    elif min_times == 1 and max_times is None:
                        seq[-1] = (f'{sub}+', False)
                    else:
                        if not sub_is_literal:
                            id = sub_rule_ids.get(sub)
                            if id is None:
                                id = self._add_rule(f'{name}-{len(sub_rule_ids) + 1}', sub)
                                sub_rule_ids[sub] = id
                            sub = id

                        seq[-1] = (
                            ' '.join(
                                ([f'"{sub[1:-1] * min_times}"'] if sub_is_literal else [sub] * min_times) +
                                ([f'{sub}?'] * (max_times - min_times) if max_times is not None else [f'{sub}*'])),
                            False
                        )
                else:
                    literal = ''
                    while i < length:
                        if pattern[i] == '\\' and i < length - 1:
                            next = pattern[i + 1]
                            if next in ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS:
                                i += 1
                                literal += pattern[i]
                                i += 1
                            else:
                                literal += pattern[i:i+2]
                                i += 2
                        elif pattern[i] == '"' and not self._raw_pattern:
                            literal += '\\"'
                            i += 1
                        elif pattern[i] not in NON_LITERAL_SET and \
                                (i == length - 1 or literal == '' or pattern[i+1] == '.' or pattern[i+1] not in NON_LITERAL_SET):
                            literal += pattern[i]
                            i += 1
                        else:
                            break
                    if literal:
                        seq.append((literal, True))

            return join_seq()

        return self._add_rule(
            name,
            to_rule(transform()) if self._raw_pattern \
                else "\"\\\"\" " + to_rule(transform()) + " \"\\\"\" space")


    def _resolve_ref(self, ref):
        ref_name = ref.split('/')[-1]
        if ref_name not in self._rules and ref not in self._refs_being_resolved:
            self._refs_being_resolved.add(ref)
            resolved = self._refs[ref]
            ref_name = self.visit(resolved, ref_name)
            self._refs_being_resolved.remove(ref)
        return ref_name

    def _generate_constant_rule(self, value):
        return self._format_literal(json.dumps(value))

    def visit(self, schema, name):
        schema_type = schema.get('type')
        schema_format = schema.get('format')
        rule_name = name + '-' if name in RESERVED_NAMES else name or 'root'

        if (ref := schema.get('$ref')) is not None:
            return self._add_rule(rule_name, self._resolve_ref(ref))

        elif 'oneOf' in schema or 'anyOf' in schema:
            return self._add_rule(rule_name, self._generate_union_rule(name, schema.get('oneOf') or schema['anyOf']))

        elif isinstance(schema_type, list):
            return self._add_rule(rule_name, self._generate_union_rule(name, [{'type': t} for t in schema_type]))

        elif 'const' in schema:
            return self._add_rule(rule_name, self._generate_constant_rule(schema['const']))

        elif 'enum' in schema:
            rule = ' | '.join((self._generate_constant_rule(v) for v in schema['enum']))
            return self._add_rule(rule_name, rule)

        elif schema_type in (None, 'object') and \
             ('properties' in schema or \
              ('additionalProperties' in schema and schema['additionalProperties'] is not True)):
            required = set(schema.get('required', []))
            properties = list(schema.get('properties', {}).items())
            return self._add_rule(rule_name, self._build_object_rule(properties, required, name, schema.get('additionalProperties')))

        elif schema_type in (None, 'object') and 'allOf' in schema:
            required = set()
            properties = []
            hybrid_name = name
            def add_component(comp_schema, is_required):
                if (ref := comp_schema.get('$ref')) is not None:
                    comp_schema = self._refs[ref]

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

            return self._add_rule(rule_name, self._build_object_rule(properties, required, hybrid_name, additional_properties=[]))

        elif schema_type in (None, 'array') and ('items' in schema or 'prefixItems' in schema):
            items = schema.get('items') or schema['prefixItems']
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space ' +
                    ' "," space '.join(
                        self.visit(item, f'{name}{"-" if name else ""}tuple-{i}')
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
            return self._visit_pattern(schema['pattern'], rule_name)

        elif schema_type in (None, 'string') and re.match(r'^uuid[1-5]?$', schema_format or ''):
            return self._add_rule(
                'root' if rule_name == 'root' else schema_format,
                PRIMITIVE_RULES['uuid']
            )

        elif schema_type in (None, 'string') and schema_format in DATE_RULES:
            for t, r in DATE_RULES.items():
                self._add_rule(t, r)
            return schema_format + '-string'

        elif (schema_type == 'object') or (len(schema) == 0):
            for n in OBJECT_RULE_NAMES:
                self._add_rule(n, PRIMITIVE_RULES[n])
            return self._add_rule(rule_name, 'object')

        else:
            assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
            # TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
            return self._add_rule(
                'root' if rule_name == 'root' else schema_type,
                PRIMITIVE_RULES[schema_type]
            )

    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], name: str, additional_properties: Union[bool, Any]):
        prop_order = self._prop_order
        # sort by position in prop_order (if specified) then by original order
        sorted_props = [kv[0] for _, kv in sorted(enumerate(properties), key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]))]

        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                fr'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}'
            )
        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]

        if additional_properties == True or isinstance(additional_properties, dict):
            sub_name = f'{name}{"-" if name else ""}additional'
            value_rule = self.visit({} if additional_properties == True else additional_properties, f'{sub_name}-value')
            prop_kv_rule_names["*"] = self._add_rule(
                f'{sub_name}-kv',
                self._add_rule('string', PRIMITIVE_RULES['string']) + f' ":" space {value_rule}'
            )
            optional_props.append("*")

        rule = '"{" space '
        rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)

        if optional_props:
            rule += ' ('
            if required_props:
                rule += ' "," space ( '

            def get_recursive_refs(ks, first_is_optional):
                [k, *rest] = ks
                kv_rule_name = prop_kv_rule_names[k]
                if k == '*':
                    res = self._add_rule(
                        f'{name}{"-" if name else ""}additional-kvs',
                        f'{kv_rule_name} ( "," space ' + kv_rule_name + ' )*'
                    )
                elif first_is_optional:
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
            )
            if required_props:
                rule += ' )'
            rule += ' )?'

        rule += ' "}" space'

        return rule

    def format_grammar(self):
        return '\n'.join(
            f'{name} ::= {rule}'
            for name, rule in sorted(self._rules.items(), key=lambda kv: kv[0])
        )


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
            properties not specified here are given lower precedence than those that are, and
            are kept in their original order from the schema. Required properties are always
            given precedence over optional properties.
        '''
    )
    parser.add_argument(
        '--allow-fetch',
        action='store_true',
        default=False,
        help='Whether to allow fetching referenced schemas over HTTPS')
    parser.add_argument(
        '--dotall',
        action='store_true',
        default=False,
        help='Whether to treat dot (".") as matching all chars including line breaks in regular expression patterns')
    parser.add_argument(
        '--raw-pattern',
        action='store_true',
        default=False,
        help='Treats string patterns as raw patterns w/o quotes (or quote escapes)')

    parser.add_argument('schema', help='file containing JSON schema ("-" for stdin)')
    args = parser.parse_args(args_in)

    if args.schema.startswith('https://'):
        url = args.schema
        import requests
        schema = requests.get(url).json()
    elif args.schema == '-':
        url = 'stdin'
        schema = json.load(sys.stdin)
    else:
        url = f'file://{args.schema}'
        with open(args.schema) as f:
            schema = json.load(f)
    converter = SchemaConverter(
        prop_order={name: idx for idx, name in enumerate(args.prop_order)},
        allow_fetch=args.allow_fetch,
        dotall=args.dotall,
        raw_pattern=args.raw_pattern)
    schema = converter.resolve_refs(schema, url)
    converter.visit(schema, '')
    print(converter.format_grammar())


if __name__ == '__main__':
    main()
