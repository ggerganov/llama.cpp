#ifdef NDEBUG
#undef NDEBUG
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

// TODO: split this to some library? common would need json.hpp
#include "../examples/server/json-schema-to-grammar.h"
#include "../examples/server/json-schema-to-grammar.cpp"

using namespace std;

static std::string trim(const std::string & source) {
    std::string s(source);
    s.erase(0,s.find_first_not_of(" \n\r\t"));
    s.erase(s.find_last_not_of(" \n\r\t")+1);
    return regex_replace(s, regex("(^|\n)[ \t]+"), "$1");
}

enum TestCaseStatus {
  SUCCESS, FAILURE
};

struct TestCase {
  TestCaseStatus expected_status;
  string name;
  string schema;
  string expected;

  void verify(const string& actual) const {
    if (trim(actual) != trim(expected)) {
      cerr << "#" << endl;
      cerr << "# Test '" << name.c_str() << "' failed." << endl;
      cerr << "#" << endl;
      cerr << schema.c_str() << endl;
      cerr << "# EXPECTED:\n" << expected.c_str() << endl;
      cerr << "# ACTUAL:\n" << actual.c_str() << endl;
      assert(false);
    }
  }
  void verify_status(TestCaseStatus status) const {
    if (status != expected_status) {
      cerr << "#" << endl;
      cerr << "# Test '" << name.c_str() << "' failed." << endl;
      cerr << "#" << endl;
      cerr << schema.c_str() << endl;
      cerr << "# EXPECTED STATUS: " << (expected_status == SUCCESS ? "SUCCESS" : "FAILURE") << endl;
      cerr << "# ACTUAL STATUS: " << (status == SUCCESS ? "SUCCESS" : "FAILURE") << endl;
      assert(false);
    }
  }
};

static void write(const string& file, const string& content) {
  ofstream f;
  f.open(file.c_str());
  f << content.c_str();
  f.close();
}

static string read(const string& file) {
  ostringstream actuals;
  actuals << ifstream(file.c_str()).rdbuf();
  return actuals.str();
}

static void test_all(const string& lang, std::function<void(const TestCase&)> runner) {
  cerr << "Testing JSON schema conversion (" << lang.c_str() << ")" << endl;
  auto test = [&](const TestCase& tc) {
    cerr << "- " << tc.name.c_str() << (tc.expected_status == FAILURE ? " (failure expected)" : "") << endl;
    runner(tc);
  };

  test({
    FAILURE,
    "unknown type",
    R"""({
      "type": "kaboom"
    })""",
    ""
  });

  test({
    FAILURE,
    "invalid type type",
    R"""({
      "type": 123
    })""",
    ""
  });

  test({
    SUCCESS,
    "exotic formats",
    R"""({
      "items": [
        { "format": "date" },
        { "format": "uuid" },
        { "format": "time" },
        { "format": "date-time" }
      ]
    })""",
    R"""(
      date ::= [0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( [0-2] [0-9] | "3" [0-1] )
      date-string ::= "\"" date "\"" space
      date-time ::= date "T" time
      date-time-string ::= "\"" date-time "\"" space
      root ::= "[" space date-string "," space uuid "," space time-string "," space date-time-string "]" space
      space ::= " "?
      time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
      time-string ::= "\"" time "\"" space
      uuid ::= "\"" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "-" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "-" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "-" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "-" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "\"" space
    )"""
  });

  test({
    SUCCESS,
    "string",
    R"""({
      "type": "string"
    })""",
    R"""(
      root ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "boolean",
    R"""({
      "type": "boolean"
    })""",
    R"""(
      root ::= ("true" | "false") space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "integer",
    R"""({
      "type": "integer"
    })""",
    R"""(
      root ::= ("-"? ([0-9] | [1-9] [0-9]*)) space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "string const",
    R"""({
      "const": "foo"
    })""",
    R"""(
      root ::= "\"foo\""
      space ::= " "?
    )"""
  });

  test({
    FAILURE,
    "non-string const",
    R"""({
      "const": 123
    })""",
    ""
  });

  test({
    FAILURE,
    "non-string enum",
    R"""({
      "enum": [123]
    })""",
    ""
  });

  test({
    SUCCESS,
    "tuple1",
    R"""({
      "prefixItems": [{ "type": "string" }]
    })""",
    R"""(
      root ::= "[" space string "]" space
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "tuple2",
    R"""({
      "prefixItems": [{ "type": "string" }, { "type": "number" }]
    })""",
    R"""(
      number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      root ::= "[" space string "," space number "]" space
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "number",
    R"""({
      "type": "number"
    })""",
    R"""(
      root ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "minItems",
    R"""({
      "items": {
        "type": "boolean"
      },
      "minItems": 2
    })""",
    R"""(
      boolean ::= ("true" | "false") space
      root ::= "[" space boolean ( "," space boolean )( "," space boolean )* "]" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "maxItems 1",
    R"""({
      "items": {
        "type": "boolean"
      },
      "maxItems": 1
    })""",
    R"""(
      boolean ::= ("true" | "false") space
      root ::= "[" space ( boolean  )? "]" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "maxItems 2",
    R"""({
      "items": {
        "type": "boolean"
      },
      "maxItems": 2
    })""",
    R"""(
      boolean ::= ("true" | "false") space
      root ::= "[" space ( boolean ( "," space boolean )? )? "]" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "min + maxItems",
    R"""({
      "items": {
        "type": ["number", "integer"]
      },
      "minItems": 3,
      "maxItems": 5
    })""",
    R"""(
      integer ::= ("-"? ([0-9] | [1-9] [0-9]*)) space
      item ::= number | integer
      number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      root ::= "[" space item ( "," space item )( "," space item )( "," space item )?( "," space item )? "]" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "regexp",
    R"""({
      "type": "string",
      "pattern": "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} and...$"
    })""",
    R"""(
      dot ::= [\U00000000-\x09\x0B\x0C\x0E-\U0010FFFF]
      root ::= ("(" root-1 root-1? root-1? ")")? root-1 root-1 root-1 "-" root-1 root-1 root-1 root-1 " and" dot dot dot
      root-1 ::= [0-9]
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "required props",
    R"""({
      "type": "object",
      "properties": {
        "a": {
          "type": "string"
        },
        "b": {
          "type": "string"
        }
      },
      "required": [
        "a",
        "b"
      ],
      "additionalProperties": false,
      "definitions": {}
    })""",
    R"""(
      a-kv ::= "\"a\"" space ":" space string
      b-kv ::= "\"b\"" space ":" space string
      root ::= "{" space a-kv "," space b-kv "}" space
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "1 optional prop",
    R"""({
      "properties": {
        "a": {
          "type": "string"
        }
      },
      "additionalProperties": false
    })""",
    R"""(
      a-kv ::= "\"a\"" space ":" space string
      root ::= "{" space  (a-kv )? "}" space
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "N optional props",
    R"""({
      "properties": {
        "a": {"type": "string"},
        "b": {"type": "string"},
        "c": {"type": "string"}
      },
      "additionalProperties": false
    })""",
    R"""(
      a-kv ::= "\"a\"" space ":" space string
      a-rest ::= ( "," space b-kv )? b-rest
      b-kv ::= "\"b\"" space ":" space string
      b-rest ::= ( "," space c-kv )?
      c-kv ::= "\"c\"" space ":" space string
      root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? "}" space
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "required + optional props",
    R"""({
      "properties": {
        "a": {"type": "string"},
        "b": {"type": "string"},
        "c": {"type": "string"},
        "d": {"type": "string"}
      },
      "required": ["a", "b"],
      "additionalProperties": false
    })""",
    R"""(
      a-kv ::= "\"a\"" space ":" space string
      b-kv ::= "\"b\"" space ":" space string
      c-kv ::= "\"c\"" space ":" space string
      c-rest ::= ( "," space d-kv )?
      d-kv ::= "\"d\"" space ":" space string
      root ::= "{" space a-kv "," space b-kv ( "," space ( c-kv c-rest | d-kv ) )? "}" space
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "additional props",
    R"""({
      "type": "object",
      "additionalProperties": {"type": "array", "items": {"type": "number"}}
    })""",
    R"""(
      additional-kv ::= string ":" space additional-value
      additional-kvs ::= additional-kv ( "," space additional-kv )*
      additional-value ::= "[" space ( number ( "," space number )* )? "]" space
      number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      root ::= "{" space  (additional-kvs )? "}" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "required + additional props",
    R"""({
      "type": "object",
      "properties": {
        "a": {"type": "number"}
      },
      "required": ["a"],
      "additionalProperties": {"type": "string"}
    })""",
    R"""(
      a-kv ::= "\"a\"" space ":" space number
      additional-kv ::= string ":" space string
      additional-kvs ::= additional-kv ( "," space additional-kv )*
      number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      root ::= "{" space a-kv ( "," space ( additional-kvs ) )? "}" space
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "optional + additional props",
    R"""({
      "type": "object",
      "properties": {
        "a": {"type": "number"}
      },
      "additionalProperties": {"type": "number"}
    })""",
    R"""(
      a-kv ::= "\"a\"" space ":" space number
      a-rest ::= additional-kvs
      additional-kv ::= string ":" space number
      additional-kvs ::= additional-kv ( "," space additional-kv )*
      number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      root ::= "{" space  (a-kv a-rest | additional-kvs )? "}" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "required + optional + additional props",
    R"""({
      "type": "object",
      "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"}
      },
      "required": ["a"],
      "additionalProperties": {"type": "number"}
    })""",
    R"""(
      a-kv ::= "\"a\"" space ":" space number
      additional-kv ::= string ":" space number
      additional-kvs ::= additional-kv ( "," space additional-kv )*
      b-kv ::= "\"b\"" space ":" space number
      b-rest ::= additional-kvs
      number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      root ::= "{" space a-kv ( "," space ( b-kv b-rest | additional-kvs ) )? "}" space
      space ::= " "?
    )"""
  });

  test({
    SUCCESS,
    "top-level $ref",
    R"""({
      "$ref": "#/definitions/MyType",
      "definitions": {
        "MyType": {
          "type": "object",
          "properties": {
            "a": {
              "type": "string"
            }
          },
          "required": [
            "a"
          ],
          "additionalProperties": false
        }
      }
    })""",
    R"""(
      MyType ::= "{" space MyType-a-kv "}" space
      MyType-a-kv ::= "\"a\"" space ":" space string
      root ::= MyType
      space ::= " "?
      string ::=  "\"" (
              [^"\\] |
              "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\"" space
    )"""
  });

  test({
    SUCCESS,
    "conflicting names",
    R"""({
      "type": "object",
      "properties": {
        "number": {
          "type": "object",
          "properties": {
            "number": {
              "type": "object",
              "properties": {
                "root": {
                  "type": "number"
                }
              },
              "required": [
                "root"
              ],
              "additionalProperties": false
            }
          },
          "required": [
            "number"
          ],
          "additionalProperties": false
        }
      },
      "required": [
        "number"
      ],
      "additionalProperties": false,
      "definitions": {}
    })""",
    R"""(
      number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
      number- ::= "{" space number-number-kv "}" space
      number-kv ::= "\"number\"" space ":" space number-
      number-number ::= "{" space number-number-root-kv "}" space
      number-number-kv ::= "\"number\"" space ":" space number-number
      number-number-root-kv ::= "\"root\"" space ":" space number
      root ::= "{" space number-kv "}" space
      space ::= " "?
    )"""
  });
}

int main() {
  test_all("Python", [](const TestCase& tc) {
    write("test-json-schema-input.tmp", tc.schema);
    tc.verify_status(std::system(
      "python ./examples/json-schema-to-grammar.py test-json-schema-input.tmp > test-grammar-output.tmp") == 0 ? SUCCESS : FAILURE);
    tc.verify(read("test-grammar-output.tmp"));
  });
  test_all("JavaScript", [](const TestCase& tc) {
    write("test-json-schema-input.tmp", tc.schema);
    tc.verify_status(std::system(
      "node ./tests/run-json-schema-to-grammar.mjs test-json-schema-input.tmp > test-grammar-output.tmp") == 0 ? SUCCESS : FAILURE);
    tc.verify(read("test-grammar-output.tmp"));
  });
  test_all("C++", [](const TestCase& tc) {
    try {
      tc.verify(json_schema_to_grammar(nlohmann::json::parse(tc.schema)));
      tc.verify_status(SUCCESS);
    } catch (const runtime_error& ex) {
      cerr << "Error: " << ex.what() << endl;
      tc.verify_status(FAILURE);
    }
  });
}
