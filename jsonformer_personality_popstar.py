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
    "agent": {
      "type": "string",
      "description": "The unique identifier for the AI robot assistant."
    },
    "knowledge": {
      "type": "string",
      "description": "The specific area of expertise or knowledge of the AI robot assistant."
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
        }
      },
      "required": ["technical_skills"]
    }
  },
  "required": ["agent", "knowledge", "context"]
}

prompt = (
    """You are a character designer generate 4 different personalities for these superheros."""
)

jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)