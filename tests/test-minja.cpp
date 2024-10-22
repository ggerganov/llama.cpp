/*
  Minimalistic Jinja templating engine for llama.cpp. C++11, no deps (single-header), decent language support but very few functions (easy to extend), just what’s needed for actual prompt templates.

  Models have increasingly complex templates (e.g. Llama 3.1, Hermes 2 Pro w/ tool_use), so we need a proper template engine to get the best out of them.

  Supports:
  - Full expression syntax
  - Statements `{{% … %}}`, variable sections `{{ … }}`, and comments `{# … #}` with pre/post space elision `{%- … -%}` / `{{- … -}}` / `{#- … -#}`
  - `if` / `elif` / `else` / `endif`
  - `for` (`recursive`) (`if`) / `else` / `endfor` w/ `loop.*` (including `loop.cycle`) and destructuring
  - `set` w/ namespaces & destructuring
  - `macro` / `endmacro`
  - Extensible filters collection: `count`, `dictsort`, `equalto`, `e` / `escape`, `items`, `join`, `joiner`, `namespace`, `raise_exception`, `range`, `reject`, `tojson`, `trim`

  Limitations:
  - Not supporting most filters & pipes. Only the ones actually used in the templates are implemented.
    https://jinja.palletsprojects.com/en/3.0.x/templates/#builtin-filters
  - No difference between none and undefined
  - Single namespace with all filters / tests / functions / macros / variables
  - No tuples (templates seem to rely on lists only)
  - No `if` expressions w/o `else` (but `if` statements are fine)
  - No `{% raw %}`, `{% block … %}`, `{% include … %}`, `{% extends … %},

  Model templates verified to work:
  - Meta-Llama-3.1-8B-Instruct
  - Phi-3.5-mini-instruct
  - Hermes-2-Pro-Llama-3-8B (default & tool_use variants)
  - Qwen2-VL-7B-Instruct, Qwen2-7B-Instruct
  - Mixtral-8x7B-Instruct-v0.1

  TODO:
  - Simplify two-pass parsing
    - Pass tokens to IfNode and such
    - Macro nested set scope = global?
      {%- macro get_param_type(param) -%}
        {%- set param_type = "any" -%}
  - Advertise in / link to https://jbmoelker.github.io/jinja-compat-tests/
*/
#include "minja.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <json.hpp>

static void assert_equals(const std::string & expected, const std::string & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

static void announce_test(const std::string & name, const minja::Options & options) {
    auto len = name.size();
    auto extract = minja::strip(name);
    extract = json(name.substr(0, std::min<size_t>(len, 50)) + (len > 50 ? " [...]" : "")).dump();
    extract = extract.substr(1, extract.size() - 2);
    std::cout << "Testing: " << extract;
    static const minja::Options default_options {};
    if (options.lstrip_blocks != default_options.lstrip_blocks)
        std::cout << " lstrip_blocks=" << options.lstrip_blocks;
    if (options.trim_blocks != default_options.trim_blocks)
        std::cout << " trim_blocks=" << options.trim_blocks;
    std::cout << std::endl << std::flush;
}

static void test_render(const std::string & template_str, const json & bindings, const minja::Options & options, const std::string & expected, const json & expected_context = {}) {
    announce_test(template_str, options);
    auto root = minja::Parser::parse(template_str, options);
    auto context = minja::Context::make(bindings);
    std::string actual;
    try {
        actual = root->render(context);
    } catch (const std::runtime_error & e) {
        actual = "ERROR: " + std::string(e.what());
    }

    assert_equals(expected, actual);

    if (!expected_context.is_null()) {
        // auto dump = context->dump();
        for (const auto & kv : expected_context.items()) {
            auto value = context->get(kv.key());
            if (value != kv.value()) {
                std::cerr << "Expected context value for " << kv.key() << ": " << kv.value() << std::endl;
                std::cerr << "Actual value: " << value.dump() << std::endl;
                std::cerr << std::flush;
                throw std::runtime_error("Test failed");
            }
        }
    }
    std::cout << "Test passed!" << std::endl << std::flush;
}

static void test_error_contains(const std::string & template_str, const json & bindings, const minja::Options & options, const std::string & expected) {
    announce_test(template_str, options);
    try {
        auto root = minja::Parser::parse(template_str, options);
        auto context = minja::Context::make(bindings);
        // auto copy = context.is_null() ? Value::object() : std::make_shared<Value>(context);
        auto actual = root->render(context);
        throw std::runtime_error("Expected error: " + expected + ", but got successful result instead: "  + actual);
    } catch (const std::runtime_error & e) {
        std::string actual(e.what());
        if (actual.find(expected) == std::string::npos) {
            std::cerr << "Expected: " << expected << std::endl;
            std::cerr << "Actual: " << actual << std::endl;
            std::cerr << std::flush;
            throw std::runtime_error("Test failed");
        }
    }
    std::cout << "  passed!" << std::endl << std::flush;
}


/*
    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -t test-minja -j && ./build/bin/test-minja
*/
int main() {
    const minja::Options lstrip_blocks {
        /* .trim_blocks = */ false,
        /* .lstrip_blocks = */ true,
        /* .keep_trailing_newline = */ false,
    };
    const minja::Options trim_blocks {
        /* .trim_blocks = */ true,
        /* .lstrip_blocks = */ false,
        /* .keep_trailing_newline = */ false,
    };
    const minja::Options lstrip_trim_blocks {
        /* .trim_blocks = */ true,
        /* .lstrip_blocks = */ true,
        /* .keep_trailing_newline = */ false,
    };

    test_render("{% set txt = 'a\\nb\\n' %}{{ txt | indent(2) }}|{{ txt | indent(2, first=true) }}", {}, {}, "a\n  b\n|  a\n  b\n");
    test_render(R"({%- if True %}        {% set _ = x %}{%- endif %}{{ 1 }})",
        {},
        lstrip_trim_blocks,
        "        1"
    );
    test_render(R"({{ "abcd"[1:-1] }})", {}, {}, "bc");
    test_render(R"({{ [0, 1, 2, 3][1:-1] }})", {}, {}, "[1, 2]");
    test_render(R"({{ "123456789" | length }})", {}, {}, "9");
    test_render(R"(  {{- 'a' -}}{{ '  ' }}{{- 'b' -}}  )", {}, {}, "a  b");
    test_render(R"(    {%- if True %}{%- endif %}{{ '        ' }}{%- for x in [] %}foo{% endfor %}end)", {}, {}, "        end");
    test_render(R"({% set ns = namespace(is_first=false, nottool=false, and_or=true, delme='') %}{{ ns.is_first }})", {}, {}, "False");
    test_render(R"({{ {} is mapping }},{{ '' is mapping }})", {}, {}, "True,False");
    test_render(R"({{ {} is iterable }},{{ '' is iterable }})", {}, {}, "True,True");
    test_render(R"({% for x in ["a", "b"] %}{{ x }},{% endfor %})", {}, {}, "a,b,");
    test_render(R"({% for x in {"a": 1, "b": 2} %}{{ x }},{% endfor %})", {}, {}, "a,b,");
    test_render(R"({% for x in "ab" %}{{ x }},{% endfor %})", {}, {}, "a,b,");
    test_render(R"({{ 'foo bar'.title() }})", {}, {}, "Foo Bar");
    test_render(R"({{ 1 | safe }})", {}, {}, "1");
    test_render(R"({{ 'abc'.endswith('bc') }},{{ ''.endswith('a') }})", {}, {}, "True,False");
    test_render(R"({{ none | selectattr("foo", "equalto", "bar") | list }})", {}, {}, "[]");
    test_render(R"({{ 'a' in {"a": 1} }},{{ 'a' in {} }})", {}, {}, "True,False");
    test_render(R"({{ 'a' in ["a"] }},{{ 'a' in [] }})", {}, {}, "True,False");
    test_render(R"({{ [{"a": 1}, {"a": 2}, {}] | selectattr("a", "equalto", 1) }})", {}, {}, R"([{'a': 1}])");
    test_render(R"({{ [{"a": 1}, {"a": 2}] | map(attribute="a") | list }})", {}, {}, "[1, 2]");
    test_render(R"({{ ["", "a"] | map("length") | list }})", {}, {}, "[0, 1]");
    test_render(R"({{ range(3) | last }})", {}, {}, "2");
    test_render(R"({% set foo = true %}{{ foo is defined }})", {}, {}, "True");
    test_render(R"({% set foo = true %}{{ not foo is defined }})", {}, {}, "False");
    test_render(R"({{ {"a": "b"} | tojson }})", {}, {}, R"({"a": "b"})");
    test_render(R"({{ {"a": "b"} }})", {}, {}, R"({'a': 'b'})");

    std::string trim_tmpl =
        "\n"
        "  {% if true %}Hello{% endif %}  \n"
        "...\n"
        "\n";
     test_render(
        trim_tmpl,
        {}, trim_blocks, "\n  Hello...\n");
     test_render(
        trim_tmpl,
        {}, {}, "\n  Hello  \n...\n");
     test_render(
        trim_tmpl,
        {}, lstrip_blocks, "\nHello  \n...\n");
     test_render(
        trim_tmpl,
        {}, lstrip_trim_blocks, "\nHello...\n");

    test_render(
        R"({%- set separator = joiner(' | ') -%}
           {%- for item in ["a", "b", "c"] %}{{ separator() }}{{ item }}{% endfor -%})",
        {}, {}, "a | b | c");
    test_render("a\nb\n", {}, {}, "a\nb");
    test_render("  {{- ' a\n'}}", {}, trim_blocks, " a\n");

    test_render(
        R"(
            {%- for x in range(3) -%}
                {%- if loop.first -%}
                    but first, mojitos!
                {%- endif -%}
                {{ loop.index }}{{ "," if not loop.last -}}
            {%- endfor -%}
        )", {}, {}, "but first, mojitos!1,2,3");
    test_render("{{ 'a' + [] | length + 'b' }}", {}, {}, "a0b");
    test_render("{{ [1, 2, 3] | join(', ') + '...' }}", {}, {}, "1, 2, 3...");
    test_render("{{ 'Tools: ' + [1, 2, 3] | reject('equalto', 2) | join(', ') + '...' }}", {}, {}, "Tools: 1, 3...");
    test_render("{{ [1, 2, 3] | join(', ') }}", {}, {}, "1, 2, 3");
    test_render("{% for i in range(3) %}{{i}},{% endfor %}", {}, {}, "0,1,2,");
    test_render("{% set foo %}Hello {{ 'there' }}{% endset %}{{ 1 ~ foo ~ 2 }}", {}, {}, "1Hello there2");
    test_render("{{ [1, False, null, True, 2, '3', 1, '3', False, null, True] | unique }}", {}, {},
        "[1, False, null, True, 2, '3']");
    test_render("{{ range(5) | length % 2 }}", {}, {}, "1");
    test_render("{{ range(5) | length % 2 == 1 }},{{ [] | length > 0 }}", {}, {}, "True,False");
    test_render(
        "{{ messages[0]['role'] != 'system' }}",
        {{"messages", json::array({json({{"role", "system"}})})}},
        {},
        "False");
    test_render(
        R"(
            {%- for x, y in [("a", "b"), ("c", "d")] -%}
                {{- x }},{{ y -}};
            {%- endfor -%}
        )", {}, {}, "a,b;c,d;");
    test_render("{{ 1 is not string }}", {}, {}, "True");
    test_render("{{ 'ab' * 3 }}", {}, {}, "ababab");
    test_render("{{ [1, 2, 3][-1] }}", {}, {}, "3");
    test_render(
        "{%- for i in range(0) -%}NAH{% else %}OK{% endfor %}",
        {}, {},
        "OK");
    test_render(
        R"(
            {%- for i in range(5) -%}
                ({{ i }}, {{ loop.cycle('odd', 'even') }}),
            {%- endfor -%}
        )", {}, {}, "(0, odd),(1, even),(2, odd),(3, even),(4, odd),");

    test_render(
        "{%- for i in range(5) if i % 2 == 0 -%}\n"
        "{{ i }}, first={{ loop.first }}, last={{ loop.last }}, index={{ loop.index }}, index0={{ loop.index0 }}, revindex={{ loop.revindex }}, revindex0={{ loop.revindex0 }}, prev={{ loop.previtem }}, next={{ loop.nextitem }},\n"
        "{% endfor -%}",
        {}, {},
        "0, first=True, last=False, index=1, index0=0, revindex=3, revindex0=2, prev=, next=2,\n"
        "2, first=False, last=False, index=2, index0=1, revindex=2, revindex0=1, prev=0, next=4,\n"
        "4, first=False, last=True, index=3, index0=2, revindex=1, revindex0=0, prev=2, next=,\n");

    test_render(
        R"(
            {%- set res = [] -%}
            {%- for c in ["<", ">", "&", '"'] -%}
                {%- set _ = res.append(c | e) -%}
            {%- endfor -%}
            {{- res | join(", ") -}}
        )", {}, {},
        R"(&lt;, &gt;, &amp;, &quot;)");
    test_render(
        R"(
            {%- set x = 1 -%}
            {%- set y = 2 -%}
            {%- macro foo(x, z, w=10) -%}
                x={{ x }}, y={{ y }}, z={{ z }}, w={{ w -}}
            {%- endmacro -%}
            {{- foo(100, 3) -}}
        )", {}, {},
        R"(x=100, y=2, z=3, w=10)");
    test_render(
        R"(
            {% macro input(name, value='', type='text', size=20) -%}
                <input type="{{ type }}" name="{{ name }}" value="{{ value|e }}" size="{{ size }}">
            {%- endmacro -%}

            <p>{{ input('username') }}</p>
            <p>{{ input('password', type='password') }}</p>)",
        {}, {}, R"(
            <p><input type="text" name="username" value="" size="20"></p>
            <p><input type="password" name="password" value="" size="20"></p>)");
    test_render(
        R"(
            {#- The values' default array should be created afresh at each call, unlike the equivalent Python function -#}
            {%- macro foo(values=[]) -%}
                {%- set _ = values.append(1) -%}
                {{- values -}}
            {%- endmacro -%}
            {{- foo() }} {{ foo() -}})",
        {}, {}, R"([1] [1])");
    test_render(R"({{ None | items | tojson }}; {{ {1: 2} | items | tojson }})", {}, {}, "[]; [[1, 2]]");
    test_render(R"({{ {1: 2, 3: 4, 5: 7} | dictsort | tojson }})", {}, {}, "[[1, 2], [3, 4], [5, 7]]");
    test_render(R"({{ {1: 2}.items() }})", {}, {}, "[[1, 2]]");
    test_render(R"({{ {1: 2}.get(1) }}; {{ {}.get(1) }}; {{ {}.get(1, 10) }})", {}, {}, "2; ; 10");
    test_render(
        R"(
            {%- for x in [1, 1.2, "a", true, True, false, False, None, [], [1], [1, 2], {}, {"a": 1}, {1: "b"}] -%}
                {{- x | tojson -}},
            {%- endfor -%}
        )", {}, {},
        R"(1,1.2,"a",true,true,false,false,null,[],[1],[1, 2],{},{"a": 1},{"1": "b"},)");
    test_render(
        R"(
            {%- set n = namespace(value=1, title='') -%}
            {{- n.value }} "{{ n.title }}",
            {%- set n.value = 2 -%}
            {%- set n.title = 'Hello' -%}
            {{- n.value }} "{{ n.title }}")", {}, {}, R"(1 "",2 "Hello")");
    test_error_contains(
        "{{ (a.b.c) }}",
        {{"a", json({{"b", {{"c", 3}}}})}},
        {},
        "'a' is not defined");
    test_render(
        "{% set _ = a.b.append(c.d.e) %}{{ a.b }}",
        json::parse(R"({
            "a": {"b": [1, 2]},
            "c": {"d": {"e": 3}}
        })"),
        {},
        "[1, 2, 3]");

    test_render(R"(
        {%- for x, y in z -%}
            {{- x }},{{ y -}};
        {%- endfor -%}
    )", {{"z", json({json({1, 10}), json({2, 20})})}}, {}, "1,10;2,20;");

    test_render(" a {{  'b' -}} c ", {}, {}, " a bc ");
    test_render(" a {{- 'b'  }} c ", {}, {}, " ab c ");
    test_render("a\n{{- 'b'  }}\nc", {}, {}, "ab\nc");
    test_render("a\n{{  'b' -}}\nc", {}, {}, "a\nbc");

    test_error_contains("{{ raise_exception('hey') }}", {}, {}, "hey");

    test_render("{{ [] is iterable }}", {}, {}, "True");
    test_render("{{ [] is not number }}", {}, {}, "True");
    test_render("{% set x = [0, 1, 2, 3] %}{{ x[1:] }}{{ x[:2] }}{{ x[1:3] }}", {}, {}, "[1, 2, 3][0, 1][1, 2]");
    test_render("{{ ' a  ' | trim }}", {}, {}, "a");
    test_render("{{ range(3) }}{{ range(4, 7) }}{{ range(0, 10, step=2) }}", {}, {}, "[0, 1, 2][4, 5, 6][0, 2, 4, 6, 8]");

    test_render(
        R"( {{ "a" -}} b {{- "c" }} )", {}, {},
        " abc ");

    test_error_contains("{% else %}", {}, {}, "Unexpected else");
    test_error_contains("{% endif %}", {}, {}, "Unexpected endif");
    test_error_contains("{% elif 1 %}", {}, {}, "Unexpected elif");
    test_error_contains("{% endfor %}", {}, {}, "Unexpected endfor");

    test_error_contains("{% if 1 %}", {}, {}, "Unterminated if");
    test_error_contains("{% for x in 1 %}", {}, {}, "Unterminated for");
    test_error_contains("{% if 1 %}{% else %}", {}, {}, "Unterminated if");
    test_error_contains("{% if 1 %}{% else %}{% elif 1 %}{% endif %}", {}, {}, "Unterminated if");

    test_render("{% if 1 %}{% elif 1 %}{% else %}{% endif %}", {}, {}, "");

    test_render(
        "{% set x = [] %}{% set _ = x.append(1) %}{{ x | tojson(indent=2) }}", {}, {},
        "[\n  1\n]");

    test_render(
        "{{ not [] }}", {}, {},
        "True");

    test_render("{{ tool.function.name == 'ipython' }}",
        json({{"tool", json({
            {"function", {{"name", "ipython"}}}
        })}}),
        {},
        "True");

    test_render(R"(
        {%- set user = "Olivier" -%}
        {%- set greeting = "Hello " ~ user -%}
        {{- greeting -}}
    )", {}, {}, "Hello Olivier");

    return 0;
}
