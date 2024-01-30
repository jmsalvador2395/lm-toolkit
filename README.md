# ml-toolkit
This package is for me to have a central and standardized library for designing experiments and collecting data.

# Installation
1. create a virtual environment using python 3.11
2. run `pip install -e .` in the project root directory
    - this allows you to execute the code by just calling `mltoolkit` when your virtual environment is enabled
3. go through the setup process for the [HuggingFace Accelerate Library](https://huggingface.co/docs/accelerate/index) by running `accelerate config`
4. to train a model, use the command `accelerate launch --no_python mltk -c <<config_path>>`
    - NOTE: you can just call `mltk -c <<config_path>>` but the program will not be using any GPUs so this would be used moreso for testing

# Configuration Files
Browse the [configuration files directory](./cfg) to get an overview of the available models and how to train them
