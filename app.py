# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# jsonformer_test.py
# SPDX-License-Identifier: MIT

import transformers
from jsonformer import Jsonformer
from lib.generator_utils import setup_tracer
import json, torch

tracer = setup_tracer()
MAX_STRING_TOKEN_LENGTH = 8192

from typing import Any, Dict

import json

def process_prompts_common(model: Any, tokenizer: Any, prompt: str, schema: Dict[str, Any]) -> str:
    if not torch.cuda.is_available():
        raise Exception("This function requires a GPU on a Linux system.")
    with tracer.start_as_current_span("process_schema"):
        with tracer.start_as_current_span("process_new_schema"):
            jsonformer = Jsonformer(model, tokenizer, schema, prompt,
                                    max_string_token_length=MAX_STRING_TOKEN_LENGTH, max_array_length=MAX_STRING_TOKEN_LENGTH, max_number_tokens=MAX_STRING_TOKEN_LENGTH)
            return jsonformer()

from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mosaicml/mpt-7b-8k"
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_name,
  device_map="auto", 
  quantization_config=nf4_config,
  trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.tie_weights()

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
    input_prompt_str = "This emote represents a catgirl face with cat ears. Generate an animation set with a name, a description, and a transition trigger based on the following schema:"
    input_schema_str = json.dumps({
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "name": {
        "type": "string",
        },
        "animation_description": {
        "type": "string",
        },
        "transition_trigger": {
        "type": "object",
        "properties": {
            "trigger_condition": {
            "type": "string",
            },
            "from_animation": {
            "type": "string",
            },
            "to_animation": {
            "type": "string",
            }
        },
        "required": ["trigger_condition", "from_animation", "to_animation"]
        }
    },
    "required": ["name", "animation_description", "transition_trigger"]
    })


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