import json, subprocess, sys, os

assert len(sys.argv) == 2
[_, pattern] = sys.argv

print(subprocess.check_output(
  [
    "python",
    os.path.join(
      os.path.dirname(os.path.realpath(__file__)),
      "json-schema-to-grammar.py"), 
    "-",
  ],
  text=True,
  input=json.dumps({
    "type": "string",
    "pattern": pattern,
  }, indent=2)))