# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
# SPDX-License-Identifier: MIT

from jsonformer import Jsonformer
from jsonformer_utils import JsonformerUtils
from jsonschema import Draft7Validator
from generator_utils import setup_tracer

tracer = setup_tracer()

MAX_STRING_TOKEN_LENGTH = 2048

import json 

def process_prompts_common(model, tokenizer, prompt, schema) -> str:
    merged_data = {}
    separated_schema = JsonformerUtils.break_apart_schema(schema)

    for new_schema in separated_schema:
        with tracer.start_as_current_span("process_new_schema"):
            jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=MAX_STRING_TOKEN_LENGTH)

        with tracer.start_as_current_span("jsonformer_generate"):
            generated_data = jsonformer()

        for key, value in generated_data.items():
            merged_data[key] = value

    return merged_data

def initialize_model_and_tokenizer():
    model_name = "philschmid/flan-ul2-20b-fp16"
    from transformers import AutoTokenizer, T5ForConditionalGeneration 
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.config.use_cache = True
    from optimum.bettertransformer import BetterTransformer
    model = BetterTransformer.transform(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
       self.model, self.tokenizer = initialize_model_and_tokenizer()

    def predict(self, 
        input_prompt: str = Input(description="Input prompt for the model"),
        input_schema: str = Input(description="Input schema for the model")) -> str:
        output = json.dumps(process_prompts_common(self.model, self.tokenizer, input_prompt, input_schema))
        return output
