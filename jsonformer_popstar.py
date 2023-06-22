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

max_length = 512

schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "description": "The entire schema must fit in 512 tokens.",
  "properties": {
    "name": {
      "type": "string",
      "description": "The name of the VTuber, which can be a combination of real or fictional words.",
      "minLength": 2,
      "maxLength": 100
    },
    "avatar": {
      "type": "object",
      "properties": {
        "appearanceType": {
          "type": "string",
          "enum": ["2D", "3D"],
          "description": "A 2D or 3D digital representation of the VTuber, often designed with unique features and characteristics."
        },
        "appearance": {
          "type": "object",
          "properties": {
            "clothing": {
              "type": "string",
              "description": "A description of the avatar's clothing."
            },
            "hair": {
              "type": "string",
              "description": "A description of the avatar's hair, including color, length, and style."
            },
            "eyes": {
              "type": "string",
              "description": "A description of the avatar's eye color and shape."
            },
            "skin": {
              "type": "string",
              "description": "A description of the avatar's skin color and texture."
            },
            "height": {
              "type": "string",
              "description": "The height of the avatar."
            },
            "weight": {
              "type": "string",
              "description": "The weight of the avatar."
            },
            "otherFeatures": {
              "type": "string",
              "description": "Any other notable features of the avatar's appearance."
            }
          },
          "description": "An object containing various properties describing the avatar's physical appearance."
        },
        "personality": {
          "type": "string",
          "description": "The personality traits of the avatar, such as mischievous, party-going, or underachiever."
        },
        "instrument": {
          "type": "string",
          "description": "The musical instrument played by the avatar in the virtual band."
        },
        "animationStyle": {
          "type": "string",
          "enum": ["hand-drawn", "computer-generated", "puppetry"],
          "description": "The style of animation used to depict the avatar, such as hand-drawn, computer-generated, or puppetry."
        },
        "voiceType": {
          "type": "string",
          "enum": ["human", "synthesized"],
          "description": "The type of voice used for the avatar, either human or synthesized."
        }
      },
      "required": ["appearanceType", "appearance", "personality", "instrument", "animationStyle", "voiceType"]
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
      "description": "A background story for the VTuber, which can include details about their origin, history, and motivations.",
      "minLength": 50,
      "maxLength": 412
    },
    "content": {
      "type": "string",
      "enum": ["gaming", "music", "art", "educational", "variety"],
      "description": "The type of content the VTuber creates, such as gaming, music, art, or educational videos."
    },
    "platform": {
      "type": "string",
      "enum": ["YouTube", "Twitch", "Other"],
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
    }
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
Sophia, the avatar creation expert, is dedicated to helping users create their perfect digital representation. Sophia believes that a well-crafted avatar can enhance one's online presence and showcase their unique personality.
Sophia is creating a new avatar Lily.
Introducing Lily Aurora Hart, a multi-talented artist hailing from the vibrant city of Melbourne, Australia. Born on August 25, 1991, Lily has made her mark as a singer, composer, and music producer, with a unique sound that fuses elements of synth-pop, indie rock, and electronic dance music. Her thought-provoking lyrics often explore themes of futurism, empowerment, and environmental consciousness. To date, she has released four studio albums, each showcasing her growth as an artist.

Lily's journey began in the early 2010s when she started sharing her music online through various platforms. She independently released her debut album, Celestial Echoes, in 2013, which caught the attention of the indie label, Solaris Records. Under their guidance, she released her sophomore album, Neon Dreams, in 2015, featuring hit singles "Stardust" and "Elysium." The album garnered critical acclaim and earned her the ARIA Award for Best Dance Release.

In 2018, Lily unveiled her third studio album, Quantum Heart, further solidifying her status as a trailblazer in the music industry. Her most recent album, Terra Nova, was released in 2021, showcasing her commitment to addressing climate change and promoting sustainability through her art.

Apart from her musical achievements, Lily has ventured into voice acting, lending her talents to the 2022 animated film, Starbound Odyssey. She also serves as a mentor on the popular reality TV show, Soundwave Showdown, where she helps aspiring musicians hone their skills and achieve their dreams.
"""]
process_prompts(input_list)