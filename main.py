# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
# SPDX-License-Identifier: MIT

from jsonformer import Jsonformer
from lib.jsonformer_utils import JsonformerUtils
from lib.generator_utils import setup_tracer
import json, torch

tracer = setup_tracer()
MAX_STRING_TOKEN_LENGTH = 2048

from typing import Any, Dict

import json

def process_prompts_common(model: Any, tokenizer: Any, prompt: str, schema: Dict[str, Any]) -> str:
    if not torch.cuda.is_available():
        raise Exception("This function requires a GPU on a Linux system.")

    merged_data = {}
    separated_schema = JsonformerUtils.break_apart_schema(schema)
    updated_prompt = prompt
    added_keys = set()

    with tracer.start_as_current_span("process_schema"):
        for new_schema in separated_schema:
            with tracer.start_as_current_span("process_new_schema"):
                jsonformer = Jsonformer(model, tokenizer, new_schema, updated_prompt, max_string_token_length=MAX_STRING_TOKEN_LENGTH)

            generated_data = {}
            with tracer.start_as_current_span("jsonformer_generate"):
                generated_data = jsonformer()

            for key, value in generated_data.items():
                if key not in merged_data:
                    merged_data[key] = value
                else:
                    if key not in added_keys:
                        updated_prompt += f" {key}: {value}"
                        added_keys.add(key)
    with tracer.start_as_current_span("process_validator"):
        final_out = Jsonformer(model, tokenizer, new_schema, merged_data, max_string_token_length=MAX_STRING_TOKEN_LENGTH)
        return final_out()


from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

from optimum.bettertransformer import BetterTransformer
model = BetterTransformer.transform(model)

from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def predict(self,
                input_prompt: str = Input(description="Input prompt for the model"),
                input_schema: str = Input(description="Input schema for the model")) -> str:
        with tracer.start_as_current_span("run_jsonformer"):
            return process_prompts_common(model, tokenizer, input_prompt, json.loads(input_schema))

def gradio_interface(input_prompt, input_schema):
    predictor = Predictor()
    predictor.setup()
    result = predictor.predict(input_prompt, input_schema)
    return result

if __name__ == "__main__":
    
    input_prompt_str ="""This emote represents a catgirl face with cat ears and a happy expression."""
    input_schema_str = """{
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "description": "A schema representing an animation with a name, a description, and a transition trigger. The animation occurs after the specified trigger.",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 3,
                "maxLength": 10,
                "description": "The name of the animation, between 3 and 10 characters long."
            },
            "animation_description": {
                "type": "string",
                "minLength": 10,
                "maxLength": 100,
                "description": "A brief description of the animation."
            },
            "transition_trigger": {
                "type": "object",
                "description": "A trigger for transitioning between animations in the animation tree. The animation occurs after this trigger.",
                "properties": {
                    "trigger_condition": {
                        "minLength": 5,
                        "maxLength": 10,
                        "type": "string",
                        "description": "The condition that must be met for the transition to occur."
                    },
                    "from_animation": {
                        "minLength": 5,
                        "maxLength": 10,
                        "type": "string",
                        "description": "The name of the animation to transition from."
                    },
                    "to_animation": {
                        "minLength": 5,
                        "maxLength": 10,
                        "type": "string",
                        "description": "The name of the animation to transition to."
                    }
                },
                "required": ["trigger_condition", "from_animation", "to_animation"]
            }
        },
        "required": ["name", "animation_description", "transition_trigger"]
    }"""
    import gradio as gr
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.components.Textbox(lines=3, label="Input Prompt"),
            gr.components.Textbox(lines=5, label="Input Schema"),
        ],
        outputs=gr.components.JSON(label="Generated JSON"),
        title="JSONFormer with Gradio",
        description="Generate JSON data based on input prompt and schema.",
        examples = [[input_prompt_str, input_schema_str]])
    iface.launch(share=True)