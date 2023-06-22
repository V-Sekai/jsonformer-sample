# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
# SPDX-License-Identifier: MIT

from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
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

def initialize_model_and_tokenizer(model_name):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU to run.")

    if "flan" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model, tokenizer = initialize_model_and_tokenizer(model_name)

input_list = ["""
    Generate a wand. It is 5 dollars.
    """]
process_prompts_common(model, tokenizer, input_list)

model_name = "philschmid/flan-ul2-20b-fp16"
model, tokenizer = initialize_model_and_tokenizer(model_name)

model.config.use_cache = True

input_list = ["""
Sophia, an avatar creation expert, helps users craft perfect digital representations to enhance their online presence. She's currently creating a new avatar, Lily Aurora Hart, a multi-talented artist from Melbourne, Australia.

Lily's 3D avatar has long purple hair, green eyes, and a futuristic outfit reflecting her passion for futurism and environmental consciousness. Her creative and ambitious personality shines through her human voice and musical talents, including playing the keyboard and synthesizer.

With the unique identifier "LAH_Music," Lily showcases skills in singing, composing, music production, and voice acting. Her backstory revolves around exploring themes of futurism, empowerment, and environmental consciousness through her lyrics.

Focusing on music content, Lily shares her work on YouTube and Twitch. Her fanbase, "Aurora Enthusiasts," supports her artistic and environmental advocacy. Collaborating with various creators, she produces engaging content and offers merchandise featuring her signature futuristic designs.

Sophia's expertise will help Lily's avatar capture her essence, connect with fans, and make a difference through art and environmental advocacy.
"""]
process_prompts_common(model, tokenizer, input_list)
