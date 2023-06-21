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

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.richconsole import RichConsoleSpanExporter

from opentelemetry.sdk.trace.export import BatchSpanProcessor

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

resource = Resource.create({ResourceAttributes.SERVICE_NAME: "service_vtuber_generator"})
tracer = trace.get_tracer(__name__)
trace.set_tracer_provider(TracerProvider(resource=resource))

span_processor = BatchSpanProcessor(RichConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

merged_data = {}
separated_schema = break_apart_schema(schema)

with tracer.start_as_current_span("break_apart_schema"):
    separated_schema = break_apart_schema(schema)

prompt = """Gura is a friendly, mischievous shark with a generally amiable personality. She has no sense of direction and often mispronounces words. Combined with her sense of laziness, this has led fans to affectionately label her a bonehead."""

logger.info(prompt)

for new_schema in separated_schema:
    with tracer.start_as_current_span("process_new_schema"):
        jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=2048)

        with tracer.start_as_current_span("jsonformer_generate"):
            generated_data = jsonformer()
            logger.info(generated_data)

        with tracer.start_as_current_span("merge_generated_data"):
            for key, value in generated_data.items():
                merged_data[key] = value

with tracer.start_as_current_span("print_merged_data"):
    logger.info(merged_data)
