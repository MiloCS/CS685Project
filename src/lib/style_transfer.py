import os

DATA_DIR = "data" # This may need to be changed on different machines

# Make sure we're in the correct directory and make sure the data directory exists
if not os.path.exists(DATA_DIR):
    os.chdir("../..") # Move up two directories because we're in src/nb and the data directory/path should be in/start at the root directory 
    assert os.path.exists(DATA_DIR), f"ERROR: DATA_DIR={DATA_DIR} not found"  # If we still can't see the data directory something is wrong

from tqdm.notebook import tqdm
import numpy as np

import torch
from src.lib.util import to_device



class StyleTransferer():

    def __init__(self, style_encoder, decoder, device):
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.device = device
    
    def build_input(self, semantic_sentence, style_sentence):
        decoder = self.decoder
        style_encoder = self.style_encoder

        positional_embeds = decoder.gpt2.transformer.wpe.weight
        token_embeds = decoder.gpt2.transformer.wte.weight

        style_encoding = style_encoder.get_style_vector([style_sentence]).squeeze(0) # (768)

        max_length = 50
        tokenized = decoder.tokenizer([semantic_sentence], return_tensors="pt", truncation=True, max_length=max_length)
        para_ids = tokenized["input_ids"] # (1, para_length)
        para_attn = tokenized["attention_mask"].squeeze(0) # (para_length)

        # para_embeds = token_embeds[para_ids].squeeze(0).detach() # (para_length, 1280)
        para_pos = positional_embeds[np.arange(0, para_ids.shape[1])].detach() # (para_length, 1280)

        target_ids = torch.tensor([[decoder.tokenizer.bos_token_id]]) # (1, 1280)
        target_pos = positional_embeds[[len(para_ids)]].detach() # (1, 1280)

        target_attn = torch.tensor([1])

        attn_mask = torch.ones(2 + len(para_attn) + len(target_attn)) # (2 + para_length + target_length)
        attn_mask[1:len(para_attn)+1] = para_attn
        attn_mask[len(para_attn)+2:] = target_attn # just one for the BOS token
        attn_mask = attn_mask

        bos_pos = positional_embeds[len(para_ids) + 1].detach()

        style_encoding = style_encoding.unsqueeze(0)
        # para_ids = para_ids.unsqueeze(0)
        para_pos = para_pos.unsqueeze(0)
        bos_pos = bos_pos.unsqueeze(0)
        # target_ids = target_ids.unsqueeze(0)
        target_pos = target_pos.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)

        x = (
            style_encoding,
            (para_ids, para_pos),
            bos_pos,
            (target_ids, target_pos),
            attn_mask.unsqueeze(0)
        )

        return x
    
    def generate(self, x, truncate=False, max_length=50):
        decoder = self.decoder
        device = self.device

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = to_device(x, device)
        decoder = decoder.to(device)
        
        # print sum of attn_mask
        

        generated_ids = []
        generated_logits = []
        with torch.no_grad():
            i = 0
            while i < max_length or not truncate:
                i += 1

                style_encoding, para, bos_pos, target, attn_mask = x
                target_ids, target_pos = target

                output = decoder(x)

                logits = output.logits[0, -1, :]
                token_id = logits.argmax()

                generated_logits.append(logits)


                # check if token_id is eos
                if token_id == decoder.tokenizer.eos_token_id:
                    break

                # add generated id 
                generated_ids.append(token_id.item())
                next_pos_embed = decoder.gpt2.transformer.wpe.weight[target_pos.shape[1] + 1]

                # update the target embedding
                target_ids = torch.cat([target_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                # update the target position
                target_pos = torch.cat([target_pos, next_pos_embed.unsqueeze(0).unsqueeze(0)], dim=1)
                # update the attention mask
                attn_mask = torch.cat([attn_mask, torch.ones(1, 1, 1).to(device)], dim=2)

                # repackage x
                x = (
                    style_encoding,
                    para,
                    bos_pos,
                    (target_ids, target_pos),
                    attn_mask
                )
            
        return generated_ids
    

    def transfer_style(self, semantic_sentence, style_sentence, truncate=True, max_length=50):
        x = self.build_input(semantic_sentence, style_sentence)
        generated_ids = self.generate(x, truncate=truncate, max_length=max_length)
        text = self.decoder.tokenizer.decode(generated_ids)
        return text