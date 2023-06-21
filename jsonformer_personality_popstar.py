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

model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = True

model = BetterTransformer.transform(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

json_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "personality": {
      "type": "string",
      "description": "The unique identifier for the personality."
    },
    "knowledge": {
      "type": "string",
      "description": "The specific area of expertise or knowledge of the personality."
    },
    "context": {
      "type": "object",
      "properties": {
        "skills": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "A list of skills possessed by the personality."
        }
      },
      "required": ["skills"]
    }
  },
  "required": ["personality", "knowledge", "context"]
}

vtuber_names = [
    "Alyssa Hartfield",
    "Cameron Oakley",
    "Evelyn Blackwood",
    "Jordan Fairchild",
    "Morgan Westfall",
    "Riley Stonebridge",
    "Taylor Greenleaf",
    "Alexis Ironwood"
]

for name in vtuber_names:
    prompt = f""""Generate VTuber {name}."""
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()
    print(generated_data)
