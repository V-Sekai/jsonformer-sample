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

model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model.config.use_cache = True

model = BetterTransformer.transform(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "The name of the VTuber, which can be a combination of real or fictional words."
    },
    "avatar": {
      "type": "string",
      "enum": ["2D", "3D"],
      "description": "A 2D or 3D digital representation of the VTuber, often designed with unique features and characteristics."
    },
    "personality": {
      "type": "object",
      "properties": {
        "uniqueIdentifier": {
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
      "required": ["uniqueIdentifier", "knowledge", "context"]
    },
    "voice": {
      "type": "string",
      "enum": ["synthesized", "human"],
      "description": "The voice of the VTuber, which can be either synthesized or provided by a human voice actor."
    },
    "backstory": {
      "type": "string",
      "description": "A fictional background story for the VTuber, which can include details about their origin, history, and motivations."
    },
    "content": {
      "type": "string",
      "enum": ["gaming", "music", "art", "educational", "variety"],
      "description": "The type of content the VTuber creates, such as gaming, music, art, or educational videos."
    },
    "platform": {
      "type": "string",
      "enum": ["YouTube", "Twitch", "Facebook Gaming", "Mixer", "DLive"],
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
    }
  },
  "required": ["name", "avatar", "personality", "voice", "backstory", "content", "platform", "fanbase", "collaborations", "merchandise"]
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
