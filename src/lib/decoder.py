import os

DATA_DIR = "data" # This may need to be changed on different machines

# Make sure we're in the correct directory and make sure the data directory exists
if not os.path.exists(DATA_DIR):
    os.chdir("../..") # Move up two directories because we're in src/nb and the data directory/path should be in/start at the root directory 
    assert os.path.exists(DATA_DIR), f"ERROR: DATA_DIR={DATA_DIR} not found"  # If we still can't see the data directory something is wrong

from tqdm.notebook import tqdm

import torch
# get Dataset class
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd

from transformers import GPT2LMHeadModel, AdamW, GPT2Tokenizer

from src.lib.decoder_dataset import DecoderDataset


class Decoder(nn.Module):


    def __init__(self, model_path="models/gpt2_large"):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_path)
        self.style_projection = nn.Linear(768, 1280)
    
    def forward(self, x):
        style_encoding, para, bos_pos, target, attn_mask = x

        para_embeds, para_pos = para
        target_embeds, target_pos = target

        # get bos embedding
        bos_embed = self.gpt2.transformer.wte.weight[self.tokenizer.bos_token_id]

        # add the positional encodings
        para_embeds += para_pos
        target_embeds += target_pos
        bos_embed  = (bos_embed + bos_pos).unsqueeze(1)

        del para_pos, target_pos, bos_pos

        # project the style encoding
        style_encoding = self.style_projection(style_encoding).unsqueeze(1)

        # concatenate style_encoding, para_embeds, bos_embed, and target_embeds
        inputs_embeds = torch.cat([style_encoding, para_embeds, bos_embed, target_embeds], dim=1)

        del x, para, target, style_encoding, para_embeds, bos_embed, target_embeds

        # get the logits
        return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn_mask)