# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
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

MAX_STRING_TOKEN_LENGTH = 2048

def process_prompts_common(model, tokenizer, input_list):
    merged_data = {}
    separated_schema = JsonformerUtils.break_apart_schema(TestUtils.get_schema())
    validator = Draft7Validator(TestUtils.get_schema())

    for prompt in input_list:
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

    return merged_data

model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_input_list():
    return ["""
    Generate a wand. It is 5 dollars.
    """]

input_list = get_input_list()
process_prompts_common(model, tokenizer, input_list)

model_name = "philschmid/flan-ul2-20b-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_name)

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This script requires a GPU to run.")

from transformers import AutoTokenizer, T5ForConditionalGeneration 

model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True) 
model.config.use_cache = True

def get_input_list_2():
    return ["""
    Generate a wand. It is 5 dollars.
    """]

input_list = get_input_list_2()
process_prompts_common(model, tokenizer, input_list)