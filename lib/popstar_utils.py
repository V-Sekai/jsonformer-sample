
class PopstarUtils:
    @staticmethod
    def get_schema():
        return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {
            "type": "string",
            "description": "VTuber's name, real or fictional words.",
            "minLength": 2,
            "maxLength": 100,
            "default": "Unknown VTuber"
            },
            "avatar": {
            "type": "object",
            "properties": {
                "appearanceType": {
                "type": "string",
                "enum": ["2D", "3D"],
                "description": "VTuber's 2D or 3D digital representation.",
                "default": "2D"
                },
                "appearance": {
                "type": "object",
                "properties": {
                    "clothing": {
                    "type": "string",
                    "description": "VTuber avatar's clothing description.",
                    "default": "Casual clothes"
                    },
                    "hair": {
                    "type": "string",
                    "description": "VTuber avatar's hair description.",
                    "default": "Short hair"
                    },
                    # "eyes": {
                    # "type": "string",
                    # "description": "VTuber avatar's eye description.",
                    # "default": "Blue eyes"
                    # },
                    "skin": {
                    "type": "string",
                    "description": "VTuber avatar's skin description.",
                    "default": "Fair skin"
                    },
                    "height": {
                    "type": "string",
                    "description": "VTuber avatar's height.",
                    "default": "Average height"
                    },
                    "weight": {
                    "type": "string",
                    "description": "VTuber avatar's weight.",
                    "default": "Average weight"
                    },
                    "otherFeatures": {
                    "type": "string",
                    "description": "VTuber avatar's other notable features.",
                    "default": "None"
                    }
                },
                "description": "VTuber avatar's physical appearance properties."
                },
                "personality": {
                "type": "string",
                "description": "VTuber avatar's personality traits.",
                "default": "Friendly"
                },
                "instrument": {
                "type": "string",
                "description": "VTuber avatar's played musical instrument.",
                "default": "Piano"
                },
                "voiceType": {
                "type": "string",
                "enum": ["human", "synthesized"],
                "description": "VTuber avatar's voice type: human or synthesized.",
                "default": "human"
                }
            },
            "required": ["appearanceType", "appearance", "personality", "instrument", "voiceType"]
            },
            "uniqueIdentifier": {
            "type": "string",
            "description": "VTuber's unique identifier.",
            "minLength": 3,
            "maxLength": 16,
            "default": "VTB000"
            },
            "skills": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of VTuber's skills.",
            "minItems": 1,
            "default": ["Singing"]
            },
            "backstory": {
            "type": "string",
            "description": "VTuber's background story, origin, history, and motivations.",
            "minLength": 50,
            "maxLength": 100,
            "default": "A virtual character with a mysterious past, seeking to entertain and inspire others."
            },
            "content": {
            "type": "string",
            "enum": ["gaming", "music", "art", "educational", "variety"],
            "description": "VTuber's content type: gaming, music, art, educational, or variety.",
            "default": "gaming"
            },
            "platform": {
            "type": "string",
            "enum": ["YouTube", "Twitch", "Other"],
            "description": "VTuber's primary content sharing platform: YouTube, Twitch, or other streaming services.",
            "default": "YouTube"
            },
            "fanbase": {
            "type": "string",
            "description": "VTuber's community of fans and supporters.",
            "default": "Virtual Enthusiasts"
            },
            "collaborations": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "VTuber's collaborations with other creators, virtual or non-virtual.",
            "default": []
            },
            "merchandise": {
            "type": "string",
            "enum": ["clothing", "accessories", "digital goods"],
            "description": "VTuber's related merchandise: clothing, accessories, or digital goods.",
            "default": "clothing"
            },
            "knowledge": {
            "type": "string",
            "description": "VTuber's specific area of expertise or knowledge.",
            "default": "Entertainment"
            }
        },
        "required": [
            "name",
            "avatar",
            "uniqueIdentifier",
            "knowledge",
            "skills",
            "backstory",
            "content",
            "platform",
            "fanbase",
            "collaborations",
            "merchandise"
        ]
        }