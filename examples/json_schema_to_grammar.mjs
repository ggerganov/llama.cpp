/*
  JSON Schema to Grammar converter (JavaScript version)

  There are C++ and Python converters w/ the same features.
  (More flags are currently exposed by the Python version)

  Usage:
    node examples/json_schema_to_grammar.mjs schema.json
    node examples/json_schema_to_grammar.mjs https://json.schemastore.org/tsconfig.json
    echo '{"type": "object"}' | node examples/json_schema_to_grammar.mjs -
*/
import { readFileSync } from "fs"
import { SchemaConverter } from "./server/public/json-schema-to-grammar.mjs"
import fs from 'fs'

const [, , file] = process.argv
let schema;
if (file === '-') {
  schema = JSON.parse(fs.readFileSync(0, 'utf8'))
} else if (file.startsWith('https://')) {
  schema = await (await fetch(file)).json()
} else {
  schema = JSON.parse(readFileSync(file, "utf8"));
}
const converter = new SchemaConverter({})
converter.visit(schema, '')
console.log(converter.formatGrammar())
