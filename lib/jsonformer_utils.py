
# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_utils.py
# SPDX-License-Identifier: MIT

import json
from typing import Any, Dict, List, Optional, Union

class JsonformerUtils:
    @staticmethod
    def break_apart_schema(schema: str, parent_required: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Breaks apart a JSON schema into smaller schemas.

        :param schema: The input JSON schema as a string.
        :param parent_required: A list of required properties from the parent schema.
        :return: A list of smaller JSON schemas.
        """
        schema_dict = {}
        if isinstance(schema, dict):
            schema_dict = schema
        else:
            schema_dict = json.loads(schema)
        parent_required = parent_required or []
        properties = schema_dict["properties"]
        required = schema_dict.get("required", [])
        result = []

        def process_items(item: Union[Dict[str, Any], Any], required: List[str]) -> Union[Dict[str, Any], Any]:
            if isinstance(item, dict) and "properties" in item:
                return JsonformerUtils.break_apart_schema(item, required)
            else:
                return item

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
