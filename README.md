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
* `nb` - Notebooks, training code, data collection, evaluation, etc. (things that get ran)

`data` - Project data
* `datasets` - datasets folder from the style_transfer_paraphrase google drive
    * `cds` - texts grouped by stle
        * `tweets` - tweet data
            * `dev.input0.bpe` - byte-pair encoded text (development set)
            * `dev.label` - labels for development set (all english_tweet)
