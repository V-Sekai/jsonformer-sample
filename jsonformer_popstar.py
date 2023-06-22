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


model_name = "tiiuae/falcon-7b-instruct"

import transformers, torch

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

system_directive = "Sophia, the avatar creation expert, is dedicated to helping users create their perfect digital representation. Sophia believes that a well-crafted avatar can enhance one's online presence and showcase their unique personality."
max_length = 2048

sequences = pipeline(
    system_directive,
    max_length=max_length,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

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
      "description": "A background story for the VTuber, which can include details about their origin, history, and motivations.",
      "minLength": 50,
      "maxLength": 100
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
                jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=max_length)
                logger.debug(f"process_new_schema: {new_schema}")

            with tracer.start_as_current_span("jsonformer_generate"):
                generated_data = jsonformer()
                logger.info(f"jsonformer_generate: {generated_data}")

            for key, value in generated_data.items():
                merged_data[key] = value

input_list = ["""

{
  "name": "Tim Cook",
  "title": "CEO of Apple Inc.",
  "born": "November 1, 1960",
  "birthplace": "Mobile, Alabama, United States",
  "education": {
    "bachelor_degree": {
      "university": "Auburn University",
      "major": "Industrial Engineering",
      "graduation_year": 1982
    },
    "master_degree": {
      "university": "Duke University",
      "major": "Master of Business Administration (MBA)",
      "graduation_year": 1988
    }
  },
  "career": [
    {
      "position": "Director of North American Fulfillment",
      "company": "IBM",
      "start_year": 1994,
      "end_year": 1997
    },
    {
      "position": "Vice President of Corporate Materials",
      "company": "Compaq",
      "start_year": 1997,
      "end_year": 1998
    },
    {
      "position": "Senior Vice President of Worldwide Operations",
      "company": "Apple Inc.",
      "start_year": 1998,
      "end_year": 2005
    },
    {
      "position": "Chief Operating Officer",
      "company": "Apple Inc.",
      "start_year": 2005,
      "end_year": 2011
    },
    {
      "position": "CEO",
      "company": "Apple Inc.",
      "start_year": 2011,
      "end_year": "present"
    }
  ]
}
"""]
process_prompts(input_list)