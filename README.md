# Installation and Running the Sample

To set up your environment and run the `jsonformer_sample.py` script, follow these steps:

## Install Micromamba

Before installing the dependencies, you need to install Micromamba. Follow the instructions for your operating system:

### Linux and macOS

```bash
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc
```

### Windows

Download the latest Micromamba executable from [here](https://micromamba.snakepit.net/api/micromamba/win-64/latest) and add it to your `PATH`.

## Install Dependencies

Open your terminal and execute the following commands to install the required dependencies:

```bash
pip install thirdparty/jsonformer.git
pip install accelerate transformers bitsandbytes optimum opentelemetry-api opentelemetry-sdk opentelemetry-exporter
micromamba install cudatoolkit -c conda-forge
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118/
```

## Create the jsonformer_sample.py Script

Create a new file named `jsonformer_sample.py` in your desired directory and add your code to it.

## Run the Script

To run the `jsonformer_sample.py` script, open your terminal, navigate to the directory containing the script, and execute the following command:

```bash
python jsonformer_sample.py
```

**Note**: Make sure you have Python installed on your system before running the script.