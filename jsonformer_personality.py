# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

# pip install thirdparty/jsonformer
# pip install accelerate transformers accelerate bitsandbytes optimum
# micromamba install cudatoolkit -c conda-forge
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

from jsonformer import Jsonformer
from optimum.bettertransformer import BetterTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "philschmid/flan-t5-xxl-sharded-fp16"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

json_schema = {
	"$schema": "http://json-schema.org/draft-07/schema#",
	"title": "OMI_personality",
	"description": "An extension for the glTF format that defines a personality for a node and an endpoint where additional information can be queried.",
	"type": "object",
	"properties": {
		"agent": {
			"type": "string",
			"description": "The name of the agent associated with the node.",
			"maxLength": 128
		},
		"personality": {
			"type": "string",
			"description": "A description of the agent's personality."
		},
		"defaultMessage": {
			"type": "string",
			"description": "The default message that the agent will send on initialization."
		}
	},
	"required": ["agent", "personality"],
}

prompt = (
    """William Henry Gates III (born October 28, 1955) is an American business magnate, investor, and philanthropist. He is best known for co-founding software giant Microsoft, along with his late childhood friend Paul Allen.[2][3] During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being its largest individual shareholder until May 2014.[4] He was a major entrepreneur of the microcomputer revolution of the 1970s and 1980s. Following this json schema."""
)

jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)
