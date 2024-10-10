# Mobile UI Design Generation using LLM and Diffusion Models

This project explores the use of Large Language Models (LLMs) and diffusion models to generate new mobile UI designs based on input descriptions. We utilize the `mrtoy/mobile-uidesign` dataset from the Hugging Face repository and build a pipeline to automate the design generation process.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the notebook and reproduce the results, install the following libraries:

```bash
# Install Hugging Face datasets and transformers
pip install datasets transformers

# Install PyTorch (for working with models later)
pip install torch

# Optionally install TensorFlow (if you prefer working with it)
pip install tensorflow

# Install diffusion models for image generation
pip install diffusers
from datasets import load_dataset
dataset = load_dataset('mrtoy/mobile-uidesign')
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from diffusers import StableDiffusionPipeline
import torch

git clone https://github.com/yourusername/mobile-ui-design-generation.git
cd mobile-ui-design-generation

# Initialize GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize stable diffusion for UI design generation
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

from PIL import Image
from IPython.display import display

# Generate mobile UI design
image = pipe("Generate a mobile UI design for a login screen").images[0]
display(image)
