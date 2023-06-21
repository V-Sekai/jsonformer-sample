# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

# pip install git+https://github.com/V-Sekai/jsonformer.git
# pip install accelerate transformers bitsandbytes optimum
# micromamba install cudatoolkit -c conda-forge
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer import Jsonformer
from optimum.bettertransformer import BetterTransformer
import torch

model_name = "ethzanalytics/RedPajama-INCITE-Instruct-7B-v0.1-sharded-bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model = BetterTransformer.transform(model)

schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "The name of the VTuber, which can be a combination of real or fictional words.",
      "minLength": 2
    },
    "avatar": {
      "type": "string",
      "enum": ["2D", "3D"],
      "description": "A 2D or 3D digital representation of the VTuber, often designed with unique features and characteristics."
    },
    "uniqueIdentifier": {
      "type": "string",
      "description": "The unique identifier for the personality.",
      "minLength": 3,
      "maxLength": 16
    },
    "skills": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "A list of skills possessed by the personality.",
      "minItems": 1
    },
    "voice": {
      "type": "string",
      "enum": ["synthesized", "human"],
      "description": "The voice of the VTuber, which can be either synthesized or provided by a human voice actor."
    },
    "backstory": {
      "type": "string",
      "enum": [
        "origin",
        "history",
        "motivations",
        "adventures",
        "relationships"
      ],
      "description": "A fictional background story for the VTuber, which can include details about their origin, history, and motivations."
    },
    "content": {
      "type": "string",
      "enum": ["gaming", "music", "art", "educational", "variety"],
      "description": "The type of content the VTuber creates, such as gaming, music, art, or educational videos."
    },
    "platform": {
      "type": "string",
      "enum": ["YouTube", "Twitch"],
      "description": "The primary platform where the VTuber shares their content, such as YouTube, Twitch, or other streaming services."
    },
    "fanbase": {
      "type": "string",
      "description": "The community of fans who follow and support the VTuber's content."
    },
    "collaborations": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Any collaborations the VTuber has done with other creators, both virtual and non-virtual."
    },
    "merchandise": {
      "type": "string",
      "enum": ["clothing", "accessories", "digital goods"],
      "description": "Any physical or digital merchandise related to the VTuber, such as clothing, accessories, or digital goods."
    },
    "knowledge": {
      "type": "string",
      "description": "The specific area of expertise or knowledge of the personality."
    },
  },
  "required": [
    "name",
    "avatar",
    "uniqueIdentifier",
    "knowledge",
    "skills",
    "voice",
    "backstory",
    "content",
    "platform",
    "fanbase",
    "collaborations",
    "merchandise"
  ]
}

def break_apart_schema(schema, parent_required=None):
    if "properties" not in schema:
        return []

    if parent_required is None:
        parent_required = []

    properties = schema["properties"]
    required = schema.get("required", [])
    result = []

    for key, value in properties.items():
        if "properties" in value or "items" in value:
            nested_required = required
            if "properties" in value:
                nested_result = break_apart_schema(value, nested_required)
            else:  # "items" in value
                nested_result = break_apart_schema(value["items"], nested_required)

            result.extend(nested_result)
            continue

        property_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                key: value
            }
        }

        if key in required or key in parent_required:
            property_schema["required"] = [key]

        result.append(property_schema)

    return result

prompt = """Gura is a friendly, mischievous shark with a generally amiable personality. She has no sense of direction and often mispronounces words. Combined with her sense of laziness, this has led fans to affectionately label her a bonehead. Follow the json schema."""

merged_data = {}
separated_schema = break_apart_schema(schema)
for new_schema in separated_schema:
    jsonformer = Jsonformer(model, tokenizer, schema, prompt, max_string_token_length=2048)
    generated_data = jsonformer()
    print(generated_data)

    for key, value in generated_data.items():
        merged_data[key] = value

print("Merged Data:")
print(merged_data)