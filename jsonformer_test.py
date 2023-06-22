# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer import Jsonformer
from lib.jsonformer_utils import JsonformerUtils
from lib.test_utils import TestUtils
from lib.popstar_utils import PopstarUtils
from jsonschema import Draft7Validator
from lib.utils import setup_logger, setup_tracer, start_heartbeat_thread

logger = setup_logger()
tracer = setup_tracer()
start_heartbeat_thread(logger, tracer)

model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_STRING_TOKEN_LENGTH = 2048
merged_data = {}
separated_schema = JsonformerUtils.break_apart_schema(TestUtils.get_schema())

validator = Draft7Validator(TestUtils.get_schema())

def process_prompts(prompts):
    for prompt in prompts:
        logger.info(f"process_prompt: {prompt}")

        for new_schema in separated_schema:
            with tracer.start_as_current_span("process_new_schema"):
                jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=MAX_STRING_TOKEN_LENGTH)
                logger.debug(f"process_new_schema: {new_schema}")

            with tracer.start_as_current_span("jsonformer_generate"):
                generated_data = jsonformer()
                logger.info(f"jsonformer_generate: {generated_data}")

            for key, value in generated_data.items():
                merged_data[key] = value

def get_input_list():
    return ["""
    Generate a wand. It is 5 dollars.
    """]

input_list = get_input_list()
process_prompts(input_list)

model_name = "philschmid/flan-ul2-20b-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_name)

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This script requires a GPU to run.")

from transformers import AutoTokenizer, T5ForConditionalGeneration 

model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True) 
model.config.use_cache = True

MAX_STRING_TOKEN_LENGTH = 2048

def process_prompts_2(prompts):
    merged_data = {}
    separated_schema = JsonformerUtils.break_apart_schema(PopstarUtils.get_schema())

    schema = TestUtils.get_schema()
    validator = Draft7Validator(schema)
    for prompt in prompts:
        logger.info(f"process_prompt: {prompt}")

        for new_schema in separated_schema:
            is_valid = validator.is_valid(new_schema)
            if not is_valid:
                logger.error(f"Invalid schema: {new_schema}")
                continue

            with tracer.start_as_current_span("process_new_schema"):
                jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=MAX_STRING_TOKEN_LENGTH)
                logger.debug(f"process_new_schema: {new_schema}")

            with tracer.start_as_current_span("jsonformer_generate"):
                generated_data = jsonformer()
                logger.info(f"jsonformer_generate: {generated_data}")

            for key, value in generated_data.items():
                merged_data[key] = value
                if not validator.is_valid({key: value}):
                    logger.error(f"Invalid data for key '{key}': {value}")

def get_input_list_2():
    return ["""
    Generate a wand. It is 5 dollars.
    """]

input_list = get_input_list_2()
process_prompts_2(input_list)
