# Text Style Transfer Project

A project for UMass CS685

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
  * `dataset_balancer.ipynb` - 

`data` - Project data
* `datasets` - datasets folder from the style_transfer_paraphrase google drive
    * `cds` - texts grouped by stle
        * `tweets` - tweet data
            * `dev.input0.bpe` - byte-pair encoded text (development set)
            * `dev.label` - labels for development set (all english_tweet)
* `vocabs`
  * `tweets.json` - json mapping tokens to integers
