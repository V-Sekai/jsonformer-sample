# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

# pip install git+https://github.com/V-Sekai/jsonformer.git
# pip install accelerate transformers bitsandbytes optimum opentelemetry-api opentelemetry-sdk opentelemetry-exporter-richconsole rich
# micromamba install cudatoolkit -c conda-forge
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

from transformers import AutoTokenizer, T5ForConditionalGeneration 
from jsonformer import Jsonformer
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,)

import logging

logger = logging.getLogger('custom_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('output.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.propagate = False

from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

tracer = trace.get_tracer(__name__)
resource = Resource.create({ResourceAttributes.SERVICE_NAME: "service_vtuber_generator"})
trace.set_tracer_provider(TracerProvider(resource=resource))

span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

import threading
import time

def send_heartbeat():
    with tracer.start_as_current_span("heartbeat") as heartbeat_span:
        while True:
            heartbeat_span.add_event("heartbeat")
            logger.info("Heartbeat event sent")
            time.sleep(20)

heartbeat_thread = threading.Thread(target=send_heartbeat)
heartbeat_thread.daemon = True
heartbeat_thread.start()


model_name = "philschmid/flan-ul2-20b-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_name)

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This script requires a GPU to run.")

model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True) 
model.config.use_cache = True

max_length = 2048

schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "VTuber's name, real or fictional words.",
      "minLength": 2,
      "maxLength": 100,
      "default": "Unknown VTuber"
    },
    "avatar": {
      "type": "object",
      "properties": {
        "appearanceType": {
          "type": "string",
          "enum": ["2D", "3D"],
          "description": "VTuber's 2D or 3D digital representation.",
          "default": "2D"
        },
        "appearance": {
          "type": "object",
          "properties": {
            "clothing": {
              "type": "string",
              "description": "VTuber avatar's clothing description.",
              "default": "Casual clothes"
            },
            "hair": {
              "type": "string",
              "description": "VTuber avatar's hair description.",
              "default": "Short hair"
            },
            "eyes": {
              "type": "string",
              "description": "VTuber avatar's eye description.",
              "default": "Blue eyes"
            },
            "skin": {
              "type": "string",
              "description": "VTuber avatar's skin description.",
              "default": "Fair skin"
            },
            "height": {
              "type": "string",
              "description": "VTuber avatar's height.",
              "default": "Average height"
            },
            "weight": {
              "type": "string",
              "description": "VTuber avatar's weight.",
              "default": "Average weight"
            },
            "otherFeatures": {
              "type": "string",
              "description": "VTuber avatar's other notable features.",
              "default": "None"
            }
          },
          "description": "VTuber avatar's physical appearance properties."
        },
        "personality": {
          "type": "string",
          "description": "VTuber avatar's personality traits.",
          "default": "Friendly"
        },
        "instrument": {
          "type": "string",
          "description": "VTuber avatar's played musical instrument.",
          "default": "Piano"
        },
        "voiceType": {
          "type": "string",
          "enum": ["human", "synthesized"],
          "description": "VTuber avatar's voice type: human or synthesized.",
          "default": "human"
        }
      },
      "required": ["appearanceType", "appearance", "personality", "instrument", "voiceType"]
    },
    "uniqueIdentifier": {
      "type": "string",
      "description": "VTuber's unique identifier.",
      "minLength": 3,
      "maxLength": 16,
      "default": "VTB000"
    },
    "skills": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of VTuber's skills.",
      "minItems": 1,
      "default": ["Singing"]
    },
    "backstory": {
      "type": "string",
      "description": "VTuber's background story, origin, history, and motivations.",
      "minLength": 50,
      "maxLength": 100,
      "default": "A virtual character with a mysterious past, seeking to entertain and inspire others."
    },
    "content": {
      "type": "string",
      "enum": ["gaming", "music", "art", "educational", "variety"],
      "description": "VTuber's content type: gaming, music, art, educational, or variety.",
      "default": "gaming"
    },
    "platform": {
      "type": "string",
      "enum": ["YouTube", "Twitch", "Other"],
      "description": "VTuber's primary content sharing platform: YouTube, Twitch, or other streaming services.",
      "default": "YouTube"
    },
    "fanbase": {
      "type": "string",
      "description": "VTuber's community of fans and supporters.",
      "default": "Virtual Enthusiasts"
    },
    "collaborations": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "VTuber's collaborations with other creators, virtual or non-virtual.",
      "default": []
    },
    "merchandise": {
      "type": "string",
      "enum": ["clothing", "accessories", "digital goods"],
      "description": "VTuber's related merchandise: clothing, accessories, or digital goods.",
      "default": "clothing"
    },
    "knowledge": {
      "type": "string",
      "description": "VTuber's specific area of expertise or knowledge.",
      "default": "Entertainment"
    }
  },
  "required": [
    "name",
    "avatar",
    "uniqueIdentifier",
    "knowledge",
    "skills",
    "backstory",
    "content",
    "platform",
    "fanbase",
    "collaborations",
    "merchandise"
  ]
}

merged_data = {}
from jsonformer_utils import JsonformerUtils

separated_schema = JsonformerUtils.break_apart_schema(schema)

def get_user_input():
    prompt = input("Enter a prompt (or type 'exit' to quit): ")
    return prompt

from jsonschema import Draft7Validator
validator = Draft7Validator(schema)

def process_prompts(prompts):
    for prompt in prompts:
        logger.info(f"process_prompt: {prompt}")

        for new_schema in separated_schema:
            with tracer.start_as_current_span("process_new_schema"):
                jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=max_length)
                logger.debug(f"process_new_schema: {new_schema}")

            with tracer.start_as_current_span("jsonformer_generate"):
                generated_data = jsonformer()
                logger.info(f"jsonformer_generate: {generated_data}")

            for key, value in generated_data.items():
                merged_data[key] = value

input_list = ["""
Sophia, an avatar creation expert, helps users craft perfect digital representations to enhance their online presence. She's currently creating a new avatar, Lily Aurora Hart, a multi-talented artist from Melbourne, Australia.

Lily's 3D avatar has long purple hair, green eyes, and a futuristic outfit reflecting her passion for futurism and environmental consciousness. Her creative and ambitious personality shines through her human voice and musical talents, including playing the keyboard and synthesizer.

With the unique identifier "LAH_Music," Lily showcases skills in singing, composing, music production, and voice acting. Her backstory revolves around exploring themes of futurism, empowerment, and environmental consciousness through her lyrics.

Focusing on music content, Lily shares her work on YouTube and Twitch. Her fanbase, "Aurora Enthusiasts," supports her artistic and environmental advocacy. Collaborating with various creators, she produces engaging content and offers merchandise featuring her signature futuristic designs.

Sophia's expertise will help Lily's avatar capture her essence, connect with fans, and make a difference through art and environmental advocacy.
"""]
process_prompts(input_list)