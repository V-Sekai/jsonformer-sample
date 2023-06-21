# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

# pip install git+https://github.com/V-Sekai/jsonformer.git
# pip install accelerate transformers accelerate bitsandbytes optimum
# micromamba install cudatoolkit -c conda-forge
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer import Jsonformer
from optimum.bettertransformer import BetterTransformer

model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = True

model = BetterTransformer.transform(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

json_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "personalities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "agent": {
            "type": "string",
            "description": "The unique identifier for the AI robot assistant."
          },
          "passion": {
            "type": "string",
            "description": "The main interest or passion of the AI robot assistant."
          },
          "dream": {
            "type": "string",
            "description": "The ultimate goal or aspiration of the AI robot assistant."
          },
          "priority": {
            "type": "string",
            "description": "The primary responsibility or focus of the AI robot assistant."
          },
          "knowledge": {
            "type": "string",
            "description": "The specific area of expertise or knowledge of the AI robot assistant."
          },
          "duty": {
            "type": "string",
            "description": "The guiding principle or approach of the AI robot assistant in fulfilling its responsibilities."
          },
          "context": {
            "type": "object",
            "properties": {
              "technical_skills": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "A list of technical skills possessed by the AI robot assistant."
              },
              "passion": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "A list of passions or interests related to the AI robot assistant's dream."
              },
              "priority": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "A list of priorities or areas of focus for the AI robot assistant."
              },
              "duty": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "A list of duties or responsibilities in the context of the AI robot assistant's role."
              }
            },
            "required": ["technical_skills", "passion", "priority", "duty"]
          }
        },
        "required": ["agent", "passion", "dream", "priority", "knowledge", "duty", "context"]
      }
    }
  },
  "required": ["personalities"]
}


prompt = (
    """You are a character designer generate 4 different personalities for these popstars."""
)

jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)