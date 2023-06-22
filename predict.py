# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
# SPDX-License-Identifier: MIT

from transformers import T5ForConditionalGeneration, AutoTokenizer
from jsonformer import Jsonformer
from jsonformer_utils import JsonformerUtils
from jsonschema import Draft7Validator
from generator_utils import setup_tracer

tracer = setup_tracer()

MAX_STRING_TOKEN_LENGTH = 2048

def process_prompts_common(model, tokenizer, input_schema_list):
    merged_data = {}

    for prompt, schema in input_schema_list:
        validator = Draft7Validator(schema)
        if not validator.is_valid(schema):
            raise ValueError(f"Invalid schema: {schema}")

        separated_schema = JsonformerUtils.break_apart_schema(schema)

        for new_schema in separated_schema:
            if not validator.is_valid(new_schema):
                raise ValueError(f"Invalid schema: {new_schema}")
            with tracer.start_as_current_span("process_new_schema"):
                jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=MAX_STRING_TOKEN_LENGTH)

            with tracer.start_as_current_span("jsonformer_generate"):
                generated_data = jsonformer()

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

import cog

class Predictor(cog.Predictor):
    def setup(self):
        self.model_name = "philschmid/flan-ul2-20b-fp16"
        self.model, self.tokenizer = initialize_model_and_tokenizer(self.model_name)

    @cog.input("prompt", type=str, help="Input prompt for the model")
    def predict(self, prompt):
        input_list = prompt
        result = process_prompts_common(input_list)
        return result
