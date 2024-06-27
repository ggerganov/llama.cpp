import { readFileSync } from "fs"
import { SchemaConverter } from "../examples/server/public/json-schema-to-grammar.mjs"

const [, , file] = process.argv
const url = `file://${file}`
const schema = JSON.parse(readFileSync(file, "utf8"));
const converter = new SchemaConverter({})
converter.visit(schema, '')
console.log(converter.formatGrammar())
