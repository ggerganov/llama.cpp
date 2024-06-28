import { readFileSync } from "fs"
import { SchemaConverter } from "../examples/server/public/json-schema-to-grammar.mjs"

const [, , file] = process.argv
let schema;
if (file.startsWith('https://')) {
  schema = await (await fetch(file)).json()
} else {
  schema = JSON.parse(readFileSync(file, "utf8"));
}
const converter = new SchemaConverter({})
converter.visit(schema, '')
console.log(converter.formatGrammar())
