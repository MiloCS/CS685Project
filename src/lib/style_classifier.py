import torch

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

class StyleEncoder:

    def __init__(self, model_path="models/style_classifier.pth", model_name="bert-base-uncased", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = BertForSequenceClassification(BertConfig.from_pretrained(
            model_name, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 11, # The number of output labels.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.))
        ))
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
    
    def get_style_vector(self, text):
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt", padding=True).input_ids.to(self.device)
            output = self.model(input_ids)
            hidden_states = output.hidden_states
            # style vector is the hidden state for the [CLS] token of the last layer
            style_vector = hidden_states[0][:, 0, :]
            return style_vector
