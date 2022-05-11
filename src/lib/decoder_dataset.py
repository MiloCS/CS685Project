import torch
# get Dataset class
from torch.utils.data import Dataset, DataLoader
from torch import nn

from tqdm.notebook import tqdm

import pandas as pd
import numpy as np

from src.lib.paraphrase_model import Paraphraser
from src.lib.style_classifier import StyleEncoder

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

    def __init__(self, df=None, batch_size=64, state_dict=None):
        if state_dict is not None:
            self.df = state_dict["df"] #
            self.style_encodings = state_dict["style_encodings"] #

            self.positional_embeds = state_dict["positional_embeds"] # 
            self.token_embeds = state_dict["token_embeds"] #

            self.para_ids = state_dict["para_ids"]
            self.para_attn = state_dict["para_attn"]

            self.target_ids = state_dict["target_ids"]
            self.target_attn = state_dict["target_attn"]
            return
        
        self.df = df
        
        paraphraser = Paraphraser()
        # Just look up these weights
        self.positional_embeds = paraphraser.model.transformer.wpe.weight
        self.token_embeds = paraphraser.model.transformer.wte.weight
        # Use the tokenizer to get the token ids and attention masks

        max_length = 50

        tokenized = paraphraser.tokenizer(list(self.df["paraphrase"]), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        self.para_ids = tokenized["input_ids"]
        self.para_attn = tokenized["attention_mask"]

        tokenized = paraphraser.tokenizer(list(self.df["text"]), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        self.target_ids = tokenized["input_ids"]
        self.target_attn = tokenized["attention_mask"]

        # get the style encodings
        style_encoder = StyleEncoder()
        self.style_encodings = []
        for i in tqdm(range(0, len(self.df), batch_size)):
            texts = list(self.df["text"][i:i+batch_size])
            self.style_encodings.append(style_encoder.get_style_vector(texts).to("cpu"))
        # concat style embeddings for all texts
        self.style_encodings = torch.cat(self.style_encodings, dim=0) # style encodings of the paraphrases in the same order as in df
        del style_encoder
        torch.cuda.empty_cache()

        # put everything on the cpu
        self.style_encodings = self.style_encodings.to("cpu").detach()
        self.positional_embeds = self.positional_embeds.to("cpu").detach()
        self.token_embeds = self.token_embeds.to("cpu").detach()
        self.para_ids = self.para_ids.to("cpu").detach()
        self.para_attn = self.para_attn.to("cpu").detach()
        self.target_ids = self.target_ids.to("cpu").detach()
        self.target_attn = self.target_attn.to("cpu").detach()

        torch.cuda.empty_cache()
    

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        
        style = self.df["label"][idx]
        random_style_encoding_idx = np.random.choice(np.where(self.df["label"] == style)[0])
        style_encoding = self.style_encodings[random_style_encoding_idx]

        para_ids = self.para_ids[idx]
        para_embed = self.token_embeds[para_ids]
        para_attn_mask = self.para_attn[idx]

        target_ids = self.target_ids[idx]
        target_embeds = self.token_embeds[target_ids]
        target_attn_mask = self.target_attn[idx].detach().clone()

        # select a random non-padding text index
        selected_idx = np.random.choice(np.where(target_attn_mask == 1)[0])
        # set the text_token_attn_mask to 0 starting at the selected index
        target_attn_mask[selected_idx:] = 0

        para_length = torch.sum(para_attn_mask).item()
        para_pos_ids = np.arange(0, len(para_ids))
        para_pos = self.positional_embeds[para_pos_ids]

        bos_pos_id = para_length + 1
        bos_pos = self.positional_embeds[bos_pos_id]

        target_pos_ids = np.arange(para_length + 2, para_length + 2 + len(target_ids))
        target_pos = self.positional_embeds[target_pos_ids]

        # attn_mask = torch.tensor([[1], para_attn_mask, [1], target_attn_mask])
        attn_mask = torch.ones(2 + len(para_attn_mask) + len(target_attn_mask))
        attn_mask[1:len(para_attn_mask)+1] = para_attn_mask
        attn_mask[len(para_attn_mask)+2:] = target_attn_mask

        label = target_ids[selected_idx]
        label_idx = len(para_ids) + 2 + selected_idx

        style_encoding = style_encoding.detach()
        para_embed = para_embed.detach()
        para_pos = para_pos.detach()
        target_embeds = target_embeds.detach()
        target_pos = target_pos.detach()
        attn_mask = attn_mask.detach()
        label = label.detach()
        bos_pos = bos_pos.detach()

        return (
            style_encoding, # Style encoding form BERT style classification of the target sentence
            (para_embed, para_pos), # Token embeddings from the paraphrased sentence with positional embeddings
            bos_pos, # Positional embedding of the BOS token
            (target_embeds, target_pos), # Token embeddings from the target sentence with positional embeddings
            attn_mask # Attention mask for the entire sequence
        ), (
            label, # Token id of the token to be predicted - index in vocab
            label_idx # Index in the sequence of the token to be predicted - index in the sequence
        )
    

    def save_state_dict(self, path):
        state = {
            "df": self.df,
            "style_encodings": self.style_encodings,
            "positional_embeds": self.positional_embeds,
            "token_embeds": self.token_embeds,
            "para_ids": self.para_ids,
            "para_attn": self.para_attn,
            "target_ids": self.target_ids,
            "target_attn": self.target_attn
        }
        torch.save(state, path)
    

    @classmethod
    def from_state_dict(cls, path):
        state = torch.load(path)
        return cls(state_dict=state)