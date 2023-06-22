# Installation and Running the Sample

Use cog.

https://github.com/replicate/cog

```bash
 sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
 pip install cog
 sudo cog predict -i prompt='Generate a wand. It is 5 dollars.' -i schema='{"$schema":"http://json-schema.org/draft-07/schema#","title":"Avatar Prop","type":"object","properties":{"id":{"description":"Unique identifier for the avatar prop."}]}'
```