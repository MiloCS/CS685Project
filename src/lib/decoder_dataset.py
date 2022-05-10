import torch
# get Dataset class
from torch.utils.data import Dataset, DataLoader
from torch import nn

from tqdm.notebook import tqdm

import pandas as pd
import numpy as np

from src.lib.paraphrase_model import Paraphraser
from src.lib.style_classifier import StyleEncoder

class DecoderDataset(Dataset):

    def __init__(self, df, batch_size=64):
        self.df = df
        
        style_encoder = StyleEncoder()
        self.style_embeds = []
        for i in tqdm(range(0, len(self.df), batch_size)):
            texts = list(self.df["text"][i:i+batch_size])
            self.style_embeds.append(style_encoder.get_style_vector(texts).to("cpu"))
        # concat style embeddings for all texts
        self.style_embeds = torch.cat(self.style_embeds, dim=0) # style embeddings of the paraphrases in the same order as in df
        del style_encoder
        torch.cuda.empty_cache()

        paraphraser = Paraphraser()
        self.para_token_embeds = []
        para_input_ids, self.para_attn_mask = paraphraser.get_input_ids_and_attention_masks(list(self.df["paraphrase"]))
        self.text_token_ids, self.text_token_attn_mask = paraphraser.get_input_ids_and_attention_masks(list(self.df["text"]))

        for i in tqdm(range(0, len(self.df), batch_size)):
            embeds = paraphraser.get_token_embeddings(para_input_ids[i:i+batch_size])
            self.para_token_embeds.append(embeds.to("cpu"))

        # concat token embeddings for all texts
        self.para_token_embeds = torch.cat(self.para_token_embeds, dim=0) # token embeddings of the paraphrases in the same order as in df
        del paraphraser
        torch.cuda.empty_cache()

        
    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        # returns (style_embed, para_token_embed, para_attn_mask), text_token_ids
        
        style = self.df["label"][idx]
        random_style_emebd_idx = np.random.choice(np.where(self.df["label"] == style)[0])
        style_embed = self.style_embeds[random_style_emebd_idx]
        para_token_embed = self.para_token_embeds[idx]
        para_attn_mask = self.para_attn_mask[idx]
        text_token_ids = self.text_token_ids[idx]

        return (style_embed, para_token_embed, para_attn_mask), text_token_ids