# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
# SPDX-License-Identifier: MIT

import os
import sys

from jsonformer import Jsonformer
from lib.jsonformer_utils import JsonformerUtils
from lib.generator_utils import setup_tracer
from transformers import AutoTokenizer, T5ForConditionalGeneration
import gradio as gr
import json, torch

tracer = setup_tracer()
MAX_STRING_TOKEN_LENGTH = 2048

def process_prompts_common(model, tokenizer, prompt, schema) -> str:
    if not torch.cuda.is_available():
        raise Exception("This function requires a GPU on a Linux system.")
    merged_data = {}
    separated_schema = JsonformerUtils.break_apart_schema(schema)
    with tracer.start_as_current_span("process_schema"):
        for new_schema in separated_schema:
                with tracer.start_as_current_span("process_new_schema"):
                    jsonformer = Jsonformer(model, tokenizer, new_schema, prompt, max_string_token_length=MAX_STRING_TOKEN_LENGTH)
                generated_data = {}
                with tracer.start_as_current_span("jsonformer_generate"):
                    generated_data = jsonformer()
                for key, value in generated_data.items():
                    merged_data[key] = value
    return merged_data


model_name = "philschmid/flan-ul2-20b-fp16"
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
model.config.use_cache = True
tokenizer = AutoTokenizer.from_pretrained(model_name)

from optimum.bettertransformer import BetterTransformer
model = BetterTransformer.transform(model)

from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def predict(self,
                input_prompt: str = Input(description="Input prompt for the model"),
                input_schema: str = Input(description="Input schema for the model")) -> str:
        output = process_prompts_common(model, tokenizer, input_prompt, input_schema)
        output = json.dumps(output)
        return output


def gradio_interface(input_prompt, input_schema):
    predictor = Predictor()
    predictor.setup()
    result = predictor.predict(input_prompt, input_schema)
    return result


if __name__ == "__main__":
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.components.Textbox(lines=3, label="Input Prompt"),
            gr.components.Textbox(lines=5, label="Input Schema"),
        ],
        outputs=gr.components.JSON(label="Generated JSON"),
        title="JSONFormer with Gradio",
        description="Generate JSON data based on input prompt and schema.",
        examples = [
            [
                "Generate a wand. It is 5 dollars.",
                """{
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "name": {
                    "type": "string"
                    },
                    "age": {
                    "type": "integer"
                    }
                }
                }"""
            ],
            [
                """{
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Avatar Prop",
                "type": "object",
                "properties": {
                    "id": {
                    "description": "Unique identifier for the avatar prop."
                    }
                }
                }""",
                """{
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "$schema": {
                    "type": "string"
                    },
                    "type": {
                    "type": "string"
                    },
                    "properties": {
                    "type": "object",
                    "propertyNames": {
                        "type": "string"
                    },
                    "additionalProperties": {
                        "type": "object",
                        "required": [
                        "type",
                        "description"
                        ],
                        "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                            "string",
                            "number",
                            "boolean",
                            "object",
                            "array"
                            ]
                        },
                        "description": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1000
                        }
                        },
                        "additionalProperties": false
                    }
                    }
                },
                "required": [
                    "type",
                    "properties"
                ],
                "additionalProperties": false
                }"""
            ]
            ]

    )
    iface.launch(share=True)