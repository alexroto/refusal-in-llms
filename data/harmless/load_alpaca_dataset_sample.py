from datasets import load_dataset
import pandas as pd

# OPTION A: Load the recommended, error-corrected version
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
