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

tokenizer = AutoTokenizer.from_pretrained(model_name)

schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Avatar Prop",
  "type": "object",
  "properties": {
    "id": {
      "type": "integer",
      "description": "Unique identifier for the avatar prop."
    },
    "name": {
      "type": "string",
      "description": "Name of the avatar prop."
    },
    "description": {
      "type": "string",
      "description": "Description of the avatar prop."
    },
    "category": {
      "type": "string",
      "description": "Category of the avatar prop."
    },
    "imageUrl": {
      "type": "string",
      "format": "uri",
      "description": "URL of the image representing the avatar prop."
    },
    "price": {
      "type": "number",
      "description": "Price of the avatar prop."
    }
  },
  "required": ["id", "name", "category", "imageUrl", "price"]
}


def break_apart_schema(schema, parent_required=None):
    """
    Breaks apart a JSON schema into smaller schemas.
    
    :param schema: The input JSON schema.
    :param parent_required: A list of required properties from the parent schema.
    :return: A list of smaller JSON schemas.
    """
    def process_items(item, required):
        if isinstance(item, dict) and "properties" in item:
            return break_apart_schema(item, required)
        else:
            return item

    if "properties" not in schema:
        return []

    parent_required = parent_required or []
    properties = schema["properties"]
    required = schema.get("required", [])
    result = []

    for key, value in properties.items():
        nested_required = value.get("required", [])

        if "properties" in value:
            result.extend(break_apart_schema(value, nested_required))
        elif "items" in value and value["items"]:
            if isinstance(value["items"], list):
                value["items"] = [process_items(item, required) for item in value["items"]]
            else:
                value["items"] = process_items(value["items"], required)
        else:
            property_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {key: value}
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

from jsonschema import Draft7Validator
validator = Draft7Validator(schema)

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
Generate a wand.
"""]
process_prompts(input_list)