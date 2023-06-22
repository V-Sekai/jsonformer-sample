# Installation and Running the Sample

Use cog.

https://github.com/replicate/cog

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
pip install cog
sudo cog build -t my-jsonformer
sudo docker run -d -p 5000:5000 --platform=linux/amd64  my-jsonformer
sudo curl http://localhost:5000/predictions -X POST --data '{"input": {"prompt": "Generate a wand. It is 5 dollars."}, "schema": "{"$schema":"http://json-schema.org/draft-07/schema#","title":"Avatar Prop","type":"object","properties":{"id":{"description":"Unique identifier for the avatar prop."}}}"}' -H "Content-Type: application/json"
```