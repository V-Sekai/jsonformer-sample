import json

class JsonformerUtils:
    @staticmethod
    def break_apart_schema(schema, parent_required=None):
        """
        Breaks apart a JSON schema into smaller schemas.

        :param schema: The input JSON schema.
        :param parent_required: A list of required properties from the parent schema.
        :return: A list of smaller JSON schemas.
        """
        def process_items(item, required):
            if isinstance(item, dict) and "properties" in item:
                return JsonformerUtils.break_apart_schema(item, required)
            else:
                return item

        if "properties" not in schema:
            return []

        parent_required = parent_required or []
        properties = schema["properties"]
        required = schema.get("required", [])
        result = []

        for key, value in properties.items():
            nested_required = value.get("required", [])

            if "properties" in value:
                result.extend(JsonformerUtils.break_apart_schema(value, nested_required))
            elif "items" in value and value["items"]:
                if isinstance(value["items"], list):
                    value["items"] = [process_items(item, required) for item in value["items"]]
                else:
                    value["items"] = process_items(value["items"], required)
            else:
                property_schema = {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {key: value}
                }

                if key in required or key in parent_required:
                    property_schema["required"] = [key]

                result.append(property_schema)

        return result
