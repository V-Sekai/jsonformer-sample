# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# vsk_jsonformer_anime_personality.py
# SPDX-License-Identifier: MIT

# pip install accelerate transformers accelerate bitsandbytes optimum
# micromamba install cudatoolkit -c conda-forge
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer import Jsonformer
from optimum.bettertransformer import BetterTransformer

model_name = "ethzanalytics/dolly-v2-12b-sharded-8bit"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = True

model = BetterTransformer.transform(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

json_schema = {
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
	"$schema": "http://json-schema.org/draft-07/schema#",
	"title": "OMI_personality",
	"description": "An extension for the glTF format that defines a personality for a node and an endpoint where additional information can be queried.",
}

prompt = (
    """Generate a new warm joyful personality following this json schema: """
)

jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)
