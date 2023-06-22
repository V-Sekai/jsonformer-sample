# Installation and Running the Sample

Use cog.

https://github.com/replicate/cog

```bash
sudo apt install docker-buildx-plugin -y
sudo cog build
sudo docker run -d -p 5000:5000 --gpus all cog-jsonformer-sample
curl http://localhost:5000/predictions -X POST -H 'Content-Type: application/json' -d '{"input": {"prompt": [["Generate a wand. It is 5 dollars.", {"$schema": "http://json-schema.org/draft-07/schema#", "title": "Avatar Prop", "type": "object", "properties": {"id": {"type": "integer", "description": "Unique identifier for the avatar prop."}, "name": {"type": "string", "description": "Name of the avatar prop."}, "description": {"type": "string", "description": "Description of the avatar prop."}, "category": {"type": "string", "description": "Category of the avatar prop."}, "imageUrl": {"type": "string", "format": "uri", "description": "URL of the image representing the avatar prop."}, "price": {"type": "number", "description": "Price of the avatar prop."}}, "required": ["id", "name", "category", "imageUrl", "price"]}]]}}'
```