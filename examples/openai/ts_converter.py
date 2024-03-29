from typing import Any, List, Set, Tuple, Union
import json

class SchemaToTypeScriptConverter:
    # TODO: comments for arguments!
    # // Get the price of a particular car model
    # type get_car_price = (_: {
    # // The name of the car model.
    # car_name: string,
    # }) => any;

    # // get the weather of a location
    # type get_weather = (_: {
    # // where to get weather.
    # location: string,
    # }) => any;
    def _desc_comment(self, schema: dict):
        desc = schema.get("description", "").replace("\n", "\n// ") if 'description' in schema else None
        return f'// {desc}\n' if desc else ''

    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], additional_properties: Union[bool, Any]):
        if additional_properties == True:
            additional_properties = {}
        elif additional_properties == False:
            additional_properties = None

        return "{" + ', '.join([
            f'{self._desc_comment(prop_schema)}{prop_name}{"" if prop_name in required else "?"}: {self.visit(prop_schema)}'
            for prop_name, prop_schema in properties
        ] + (
            [f"{self._desc_comment(additional_properties) if additional_properties else ''}[key: string]: {self.visit(additional_properties)}"]
            if additional_properties is not None else []
        )) + "}"

    def visit(self, schema: dict):
        def print_constant(v):
            return json.dumps(v)

        schema_type = schema.get('type')
        schema_format = schema.get('format')

        if 'oneOf' in schema or 'anyOf' in schema:
            return '|'.join(self.visit(s) for s in schema.get('oneOf') or schema.get('anyOf'))

        elif isinstance(schema_type, list):
            return '|'.join(self.visit({'type': t}) for t in schema_type)

        elif 'const' in schema:
            return print_constant(schema['const'])

        elif 'enum' in schema:
            return '|'.join((print_constant(v) for v in schema['enum']))

        elif schema_type in (None, 'object') and \
              ('properties' in schema or \
              ('additionalProperties' in schema and schema['additionalProperties'] is not True)):
            required = set(schema.get('required', []))
            properties = list(schema.get('properties', {}).items())
            return self._build_object_rule(properties, required, schema.get('additionalProperties'))

        elif schema_type in (None, 'object') and 'allOf' in schema:
            required = set()
            properties = []
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

            return self._build_object_rule(properties, required, additional_properties=[])

        elif schema_type in (None, 'array') and ('items' in schema or 'prefixItems' in schema):
            items = schema.get('items') or schema['prefixItems']
            if isinstance(items, list):
                return '[' + ', '.join(self.visit(item) for item in items) + '][]'
            else:
                return self.visit(items) + '[]'

        elif schema_type in (None, 'string') and schema_format == 'date-time':
            return 'Date'

        elif (schema_type == 'object') or (len(schema) == 0):
            return 'any'

        else:
            return 'number' if schema_type == 'integer' else schema_type
