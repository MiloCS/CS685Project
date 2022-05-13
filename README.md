# Text Style Transfer Project

A project for UMass CS685

## File Structure:

`src` - Project source code
* `lib` - Library code, functions, classes, etc. (things that get imported)
  * `bpe_parser.py`- functions to parse BPE data to text
  * `decoder_dataset.py` - PyTorch Dataset class for decoder. Gets all the components of the somewhat convoluted input sequence of the decoder
  * `decoder.py` - Combines the parts from the DecoderDataset into inputs to a gpt2 model
  * `paraphrase_model.py` - Loads up the pre-trained paraphrase model, has methods to do paraphrasing
  * `plotting.py` - Used like one time to make some hexbin plots of semantic embeddings that we didn't end up using
  * `style_classifier.py` - Loads the BERT style classifier and can be used to get a sentence encoding
  * `style_transfer.py` - Can process two sentences into inputs the decoder can use. Autoregressivley genertates text based on encoders output
* `nb` - Notebooks, training code, data collection, evaluation, etc. (things that get ran)
  * `semantic_encoder`
    * Used a pre-trained model to encoder the data and plot some graphs of the encodings  
  * `bpe_reader.ipynb` - proof of concept reading data into numpy arrays from the bpe files
  * `vocab_decoder.ipynb` - Used to join decode and join the cds bpe data into a single natural language csv file
  * `dataset_balancer.ipynb` - Balances dataset so there are the same number of samples for each class
  * `decoder_dataset_test.ipynb` - Notebook to test the decoder dataset
  * `decoder_test.ipynb` - Notebook to test the decoder, the functionality not the performance
  * `get_data_for_eval.ipynb` - Uses the defunct decoder to do style transfer on the test data
  * `paraphrase_test.ipynb` - Loads the pre-trainde paraphrase model and makes sure it works as expected
  * `style_classifier_explore.ipynb` - Loads the style classifier and was used to figure out how to extract the style encoding form the classifier
  * `style_transfer_test.ipynb` - Tests the style transfer class
  * `train_decoder.ipynb` - Trains the decoder but probably has at least one issue that causes the decoder to spam one token per sample
  * `vocab_decoder.ipynb` - Basically just tokenizer.decode() but it's an entire notebook, this was before I understood tokenizer.decode()

## Project Overview:

Semantic Encoder
* Styled Sentence
* Paraphrase model - Pretrained paraphrase model 🤗
* Unstyled sentence
* Entailment BERT - Pretrained entailment BERT 🤗
* Semantic vector

Style Encoder
* Styled sentence
* Multiple augmented sentence representations
* Multiple style classifiers - _maybe_ pretrained classifier models 🤗❓
* Multiple style vectors
* Concatentation
* Style vector

Decoder
* Semantic and style vectors
* Given to GPT2 as word vectors - Pretrained GPT2 🤗
* Decode into original sentence

Style Transfer
* Two styled sentences
* Encode semantic and style vectors
* Swap style vectors
* Give swapped vector pairs to GPT2-decoder
* Decode into style transfered sentences



