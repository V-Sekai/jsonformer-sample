# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

# pip install git+https://github.com/V-Sekai/jsonformer.git
# pip install accelerate transformers bitsandbytes optimum
# micromamba install cudatoolkit -c conda-forge
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

from jsonformer import Jsonformer
from optimum.bettertransformer import BetterTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "philschmid/flan-t5-xxl-sharded-fp16"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
model.config.use_cache = True

model = BetterTransformer.transform(model)

name_schema = {
  "type": "string",
  "description": "The name of the VTuber, which can be a combination of real or fictional words.",
  "minLength": 2
}

avatar_schema = {
  "type": "string",
  "enum": ["2D", "3D"],
  "description": "A 2D or 3D digital representation of the VTuber, often designed with unique features and characteristics."
}

uniqueIdentifier_schema = {
  "type": "string",
  "description": "The unique identifier for the personality.",
  "minLength": 3,
  "maxLength": 16
}

knowledge_schema = {
  "type": "string",
  "description": "The specific area of expertise or knowledge of the personality.",
}

skills_schema = {
  "type": "array",
  "items": {
    "type": "string"
  },
  "description": "A list of skills possessed by the personality.",
  "minItems": 1
}

voice_schema = {
  "type": "string",
  "enum": ["synthesized", "human"],
  "description": "The voice of the VTuber, which can be either synthesized or provided by a human voice actor."
}

backstory_schema = {
  "type": "string",
  "enum": [
    "origin",
    "history",
    "motivations",
    "adventures",
    "relationships"
  ],
  "description": "A fictional background story for the VTuber, which can include details about their origin, history, and motivations.",
}

content_schema = {
  "type": "string",
  "enum": ["gaming", "music", "art", "educational", "variety"],
  "description": "The type of content the VTuber creates, such as gaming, music, art, or educational videos."
}

platform_schema = {
  "type": "string",
  "enum": ["YouTube", "Twitch"],
  "description": "The primary platform where the VTuber shares their content, such as YouTube, Twitch, or other streaming services."
}

fanbase_schema = {
  "type": "string",
  "description": "The community of fans who follow and support the VTuber's content.",
}

collaborations_schema = {
  "type": "array",
  "items": {
    "type": "string"
  },
  "description": "Any collaborations the VTuber has done with other creators, both virtual and non-virtual."
}

merchandise_schema = {
  "type": "string",
  "enum": ["clothing", "accessories", "digital goods"],
  "description": "Any physical or digital merchandise related to the VTuber, such as clothing, accessories, or digital goods."
}

subschemas = {
    "name": name_schema,
    "avatar": avatar_schema,
    "uniqueIdentifier": uniqueIdentifier_schema,
    "knowledge": knowledge_schema,
    "skills": skills_schema,
    "voice": voice_schema,
    "backstory": backstory_schema,
    "content": content_schema,
    "platform": platform_schema,
    "fanbase": fanbase_schema,
    "collaborations": collaborations_schema,
    "merchandise": merchandise_schema
}

prompt = """Gura is a friendly, mischievous shark with a generally amiable personality. She has no sense of direction and often mispronounces words. Combined with her sense of laziness, this has led fans to affectionately label her a bonehead."""

def generate_data(property_name, schema):
    if schema["type"] == "array":
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                property_name: {
                    "type": "array",
                    "items": schema["items"]
                }
            },
            "required": [property_name]
        }
    else:
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                property_name: schema
            },
            "required": [property_name]
        }

    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_string_token_length=2048)
    generated_data = jsonformer()

    return {property_name: generated_data[property_name]}

merged_data = {}
for property_name, schema in subschemas.items():
    generated_data = generate_data(property_name, schema)
    print(generated_data)

    for key, value in generated_data.items():
        merged_data[key] = value

print("Merged Data:")
print(merged_data)


