# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

# pip install git+https://github.com/V-Sekai/jsonformer.git
# pip install accelerate transformers bitsandbytes optimum opentelemetry-api opentelemetry-sdk opentelemetry-exporter-richconsole rich
# micromamba install cudatoolkit -c conda-forge
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer import Jsonformer
from optimum.bettertransformer import BetterTransformer

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


model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

model = BetterTransformer.transform(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

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
      "description": "A fictional background story for the VTuber, which can include details about their origin, history, and motivations."
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
        if "properties" in value:
            nested_required = required
            nested_result = break_apart_schema(value, nested_required)
            result.extend(nested_result)
            continue

        if "items" in value and value["items"]:
            if isinstance(value["items"], dict) and "properties" in value["items"]:
                value["items"] = break_apart_schema(value["items"], required)
            elif isinstance(value["items"], list):
                value["items"] = [break_apart_schema(item, required) for item in value["items"] if isinstance(item, dict) and "properties" in item]

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


merged_data = {}

separated_schema = break_apart_schema(schema)


def get_user_input():
    prompt = input("Enter a prompt (or type 'exit' to quit): ")
    return prompt


def process_prompts(prompts):
    for prompt in prompts:
        logger.info(f"process_prompt: {prompt}")

        for new_schema in separated_schema:
            with tracer.start_as_current_span("process_new_schema"):
                jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=2048)
                logger.debug(f"process_new_schema: {new_schema}")

            with tracer.start_as_current_span("jsonformer_generate"):
                generated_data = jsonformer()
                logger.info(f"jsonformer_generate: {generated_data}")
        for key, value in generated_data.items():
            merged_data[key] = value

input_list = ["""
{
  "name": "Sir Isaac Newton",
  "title": "Mathematician, Physicist, Astronomer, and Author",
  "born": "December 25, 1642",
  "died": "March 20, 1726",
  "birthplace": "Woolsthorpe, Lincolnshire, England",
  "education": {
    "school": "The King's School, Grantham",
    "university": "Trinity College, Cambridge"
  },
  "career": [
    {
      "position": "Lucasian Professor of Mathematics",
      "start_year": 1669,
      "end_year": 1702
    },
    {
      "position": "Warden and Master of the Royal Mint",
      "start_year": 1696,
      "end_year": 1726
    },
    {
      "position": "President of the Royal Society",
      "start_year": 1703,
      "end_year": 1726
    }
  ],
  "contributions": [
    "Laws of Motion",
    "Universal Gravitation",
    "Calculus",
    "Optics"
  ]
}
"""]
process_prompts(input_list)