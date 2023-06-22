class TestUtils:
    @staticmethod
    def get_schema():
        return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Avatar Prop",
        "type": "object",
        "properties": {
            "id": {
            "type": "integer",
            "description": "Unique identifier for the avatar prop."
            },
            "name": {
            "type": "string",
            "description": "Name of the avatar prop."
            },
            "description": {
            "type": "string",
            "description": "Description of the avatar prop."
            },
            "category": {
            "type": "string",
            "description": "Category of the avatar prop."
            },
            "imageUrl": {
            "type": "string",
            "format": "uri",
            "description": "URL of the image representing the avatar prop."
            },
            "price": {
            "type": "number",
            "description": "Price of the avatar prop."
            }
        },
        "required": ["id", "name", "category", "imageUrl", "price"]
        }