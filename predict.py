# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
# SPDX-License-Identifier: MIT

from transformers import T5ForConditionalGeneration, AutoTokenizer
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

def process_prompts_common(model, tokenizer, input_schema_list):
    merged_data = {}

    for prompt, schema in input_schema_list:
        logger.info(f"process_prompt: {prompt}")
        separated_schema = JsonformerUtils.break_apart_schema(schema)
        validator = Draft7Validator(schema)

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

def initialize_model_and_tokenizer():
    model_name = "philschmid/flan-ul2-20b-fp16"
    model.config.use_cache = True

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU to run.")

    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

input_schema_list = [("""
Sophia, an avatar creation expert, helps users craft perfect digital representations to enhance their online presence. She's currently creating a new avatar, Lily Aurora Hart, a multi-talented artist from Melbourne, Australia.

Lily's 3D avatar has long purple hair, green eyes, and a futuristic outfit reflecting her passion for futurism and environmental consciousness. Her creative and ambitious personality shines through her human voice and musical talents, including playing the keyboard and synthesizer.

With the unique identifier "LAH_Music," Lily showcases skills in singing, composing, music production, and voice acting. Her backstory revolves around exploring themes of futurism, empowerment, and environmental consciousness through her lyrics.

Focusing on music content, Lily shares her work on YouTube and Twitch. Her fanbase, "Aurora Enthusiasts," supports her artistic and environmental advocacy. Collaborating with various creators, she produces engaging content and offers merchandise featuring her signature futuristic designs.

Sophia's expertise will help Lily's avatar capture her essence, connect with fans, and make a difference through art and environmental advocacy.
""", PopstarUtils.get_schema()),
("""
    Generate a wand. It is 5 dollars.
    """, TestUtils.get_schema())
]

process_prompts_common(input_schema_list)

import cog

class JsonformerTest(cog.Predictor):
    def setup(self):
        self.model_name = "philschmid/flan-ul2-20b-fp16"
        self.model, self.tokenizer = initialize_model_and_tokenizer(self.model_name)

    @cog.input("prompt", type=str, help="Input prompt for the model")
    def predict(self, prompt):
        input_list = [prompt]
        result = process_prompts_common(input_list)
        return result
