import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import os

class Paraphraser:

    def __init__(self, model_path="models/gpt2_large", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)


    def get_token_embeddings(self, input_ids):
        """ Returns the embeddings of the tokens in the input_ids with positional embeddings."""
        embeddings = self.model.transformer.wte.weight
        position_embeddings = self.model.transformer.wpe.weight
        seq_len = input_ids.shape[1]
        return embeddings[input_ids] + position_embeddings[torch.arange(seq_len).repeat(input_ids.shape[0], 1).to(self.device)]
        


    def get_input_ids_and_attention_masks(self, texts):
        """See method name"""
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        return input_ids, attention_mask


    def paraphrase(self, input_ids, attention_mask):
        """ Paraphrases a given input_ids and attention_mask.

        Args:
            input_ids (batch_size, seq_length): [[word_id, word_id, pad, pad, pad] ]
            attention_mask (batch_size, seq_length): [[1, 1, 0, 0, 0]]

        Returns:
            generated: [[word_id, word_id, word_id, eos, pad, pad]]
            attention_mask: [[1, 1, 1, 0, 0, 0]]
        """
        batch_size = input_ids.shape[0]
        max_generate = int(input_ids.shape[1]  * 3)
        
        bos_token = self.tokenizer.bos_token_id
        
        # how I do this next part might make the positional encodings a little weird... but I don't know how else to do it
        # Add the BOS token to the input ids
        bos_tensor = torch.tensor([bos_token]).to(self.device)
        # repeat bos_tensor batch_size times
        bos_tensor = bos_tensor.repeat(batch_size, 1)
        input_ids = torch.cat((input_ids, bos_tensor), dim=1)

        # add the BOS token to the attention mask
        bos_attention = torch.tensor([1]).to(self.device)
        att_tensor = bos_attention.repeat(batch_size, 1)
        attention_mask = torch.cat((attention_mask, att_tensor), dim=1)

        # generate until the model generates the EOS token or reaches max_generate
        generated = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            max_length=max_generate,
        )

        # select the tokens generated after the BOS token and before the EOS token
        generated = generated[:, input_ids.shape[1]:-1]
        attention_mask = torch.where(torch.logical_and(generated != self.tokenizer.pad_token_id, generated != self.tokenizer.eos_token_id), 1, 0)

        return generated, attention_mask
    


        
# class Embedder