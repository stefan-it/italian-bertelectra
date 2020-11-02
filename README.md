# 🇮🇹 Italian BERTELECTRA models

In this repository we monitor all experiments for our trained [DBMDZ models](https://github.com/dbmdz/berts)
for Italian. It includes both BERT and ELECTRA models and we use 🤗 Transformers library to fine-tune models.

Made with 🤗 and ❤️ from Munich.

# Changelog

* 02.11.2020: Initial version and public release of Italian XXL ELECTRA model.

# Training

The source data for the Italian BERT model consists of a recent Wikipedia dump and
various texts from the [OPUS corpora](http://opus.nlpl.eu/) collection. The final
training corpus has a size of 13GB and 2,050,057,573 tokens.

For sentence splitting, we use NLTK (faster compared to spacy).
Our cased and uncased models are training with an initial sequence length of 512
subwords for ~2-3M steps.

For the XXL Italian models, we use the same training data from OPUS and extend
it with data from the Italian part of the [OSCAR corpus](https://traces1.inria.fr/oscar/).
Thus, the final training corpus has a size of 81GB and 13,138,379,147 tokens.

Note: Unfortunately, a wrong vocab size was used when training the XXL models.
This explains the mismatch of the "real" vocab size of 31,102, compared to the
vocab size specified in `config.json`. However, the model is working and all
evaluations were done under those circumstances.
See [this issue](https://github.com/dbmdz/berts/issues/7) for more information.

The Italian ELECTRA model was trained on the "XXL" corpus for 1M steps in total using a batch
size of 128. We pretty much following the ELECTRA training procedure as used for
[BERTurk](https://github.com/stefan-it/turkish-bert/tree/master/electra). The ELECTRA model uses
the same vocab as the XXL BERT model; but this time we use the correct vocab size in the `config.json`
file ;)

# Experiments

We use the awesome 🤗 Transformers library for all fine-tuning experiments.

Please star and watch [Transformers](https://github.com/huggingface/transformers) on GitHub!

All JSON-based configuration files for our experiments can be found in the
[configuration](https://github.com/stefan-it/italian-bertelectra/tree/main/configs) folder
in this repository. To replicate the results, just clone the latest version of Transforms, `cd`
into the `examples/token-classification` folder and run `python3 run_ner.py <configuration.json>`.

## PoS Tagging

### Italian-ISDT

Description:

> The Italian corpus annotated according to the UD annotation scheme was obtained by conversion
> from ISDT (Italian Stanford Dependency Treebank), released for the dependency parsing shared
> task of Evalita-2014 (Bosco et al. 2014).

Details:

* [Italian-ISDT Repository](https://github.com/UniversalDependencies/UD_Italian-ISDT)
* Commit: `f20fa2b`

Results (Development set)

| Model                                         | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| --------------------------------------------- | ----- | ----- | ----- | ----- | ----- | -------------- |
| `dbmdz/bert-base-italian-cased`               | 98.56 | 98.64 | 98.65 | 98.74 | 98.62 | 98.64 ± 0.06
| `dbmdz/bert-base-italian-uncased`             | 98.17 | 98.25 | 98.23 | 98.21 | 98.24 | 98.22 ± 0.03
| `dbmdz/bert-base-italian-xxl-cased`           | 98.52 | 98.63 | 98.76 | 98.70 | 98.63 | 98.65 ± 0.08
| `dbmdz/bert-base-italian-xxl-uncased`         | 98.38 | 98.33 | 98.35 | 98.41 | 98.30 | 98.35 ± 0.04
| `bert-base-multilingual-cased`                | 98.50 | 98.54 | 98.49 | 98.54 | 98.43 | 98.50 ± 0.04
| `bert-base-multilingual-uncased`              | 98.24 | 98.17 | 98.22 | 98.27 | 98.23 | 98.23 ± 0.03
| `xlm-roberta-base`                            | 98.63 | 98.61 | 98.66 | 98.60 | 98.62 | 98.62 ± 0.02
| `electra-base-italial-xxl-cased` (1M)         | 98.72 | 98.75 | 98.78 | 98.68 | 98.76 | **98.74** ± 0.03

Results (Test set)

| Model                                         | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| --------------------------------------------- | ----- | ----- | ----- | ----- | ----- | --------------- |
| `dbmdz/bert-base-italian-cased`               | 98.81 | 98.88 | 98.93 | 98.77 | 98.79 | 98.84 ± 0.06
| `dbmdz/bert-base-italian-uncased`             | 98.65 | 98.54 | 98.66 | 98.62 | 98.61 | 98.62 ± 0.04
| `dbmdz/bert-base-italian-xxl-cased`           | 98.93 | 98.85 | 98.92 | 98.89 | 98.89 | **98.90** ± 0.03
| `dbmdz/bert-base-italian-xxl-uncased`         | 98.72 | 98.65 | 98.73 | 98.75 | 98.78 | 98.73 ± 0.04
| `bert-base-multilingual-cased`                | 98.79 | 98.71 | 98.60 | 98.72 | 98.64 | 98.69 ± 0.07
| `bert-base-multilingual-uncased`              | 98.57 | 98.51 | 98.49 | 98.47 | 98.44 | 98.50 ± 0.04
| `xlm-roberta-base`                            | 98.86 | 98.79 | 98.78 | 98.77 | 98.76 | 98.79 ± 0.04
| `electra-base-italial-xxl-cased` (1M)         | 98.84 | 98.93 | 98.82 | 98.88 | 98.92 | 98.88 ± 0.04

### Italian-PoSTWITA

Description:

> PoSTWITA-UD has been created by enriching the dataset used for the EVALITA 2016 task of Part-of-Speech tagging
> of Social Media (see (Bosco et al. 2016)). The original corpus consists of 6,438 tweets of the development set
> (114,967 tokens) and 300 tweets of the test set (4,759 tokens), annotated at PoS level only. The conversion and
> syntactic annotation process was carried out through alternating steps of automatic scripting and manual revision,
> and finally with some out-of-domain parsing experiments. Parsing results also underwent a manual revision by two
> independent annotators.

Details:

* [Italian-PoSTWITA Repository](https://github.com/UniversalDependencies/UD_Italian-PoSTWITA)
* Commit: `662b235`

Results (Development set)

| Model                                         | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| --------------------------------------------- | ----- | ----- | ----- | ----- | ----- | -------------- |
| `dbmdz/bert-base-italian-cased`               | 96.83 | 96.80 | 96.93 | 96.87 | 96.89 | 96.86 ± 0.05
| `dbmdz/bert-base-italian-uncased`             | 96.82 | 96.80 | 96.69 | 96.76 | 96.69 | 96.75 ± 0.05
| `dbmdz/bert-base-italian-xxl-cased`           | 97.34 | 97.37 | 97.35 | 97.31 | 97.22 | 97.32 ± 0.05
| `dbmdz/bert-base-italian-xxl-uncased`         | 96.96 | 96.96 | 97.12 | 96.98 | 97.02 | 97.01 ± 0.06
| `bert-base-multilingual-cased`                | 95.96 | 95.92 | 95.98 | 95.94 | 95.92 | 95.94 ± 0.02
| `bert-base-multilingual-uncased`              | 96.06 | 96.17 | 96.28 | 96.23 | 96.25 | 96.20 ± 0.08
| `xlm-roberta-base`                            | 96.81 | 96.73 | 96.87 | 96.80 | 96.74 | 96.79 ± 0.05
| `electra-base-italian-xxl-cased` (1M)         | 97.35 | 97.26 | 97.33 | 97.36 | 97.47 | **97.35** ± 0.07

Results (Test set)

| Model                                         | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| --------------------------------------------- | ----- | ----- | ----- | ----- | ----- | -------------- |
| `dbmdz/bert-base-italian-cased`               | 96.93 | 97.03 | 97.05 | 97.12 | 97.11 | 97.05 ± 0.07
| `dbmdz/bert-base-italian-uncased`             | 96.73 | 96.71 | 96.58 | 96.60 | 96.58 | 96.64 ± 0.07
| `dbmdz/bert-base-italian-xxl-cased`           | 97.15 | 97.17 | 97.34 | 97.05 | 97.12 | 97.17 ± 0.10
| `dbmdz/bert-base-italian-xxl-uncased`         | 96.95 | 97.12 | 96.96 | 97.02 | 97.01 | 97.01 ± 0.06
| `bert-base-multilingual-cased`                | 96.17 | 96.15 | 96.09 | 96.15 | 96.07 | 96.13 ± 0.04
| `bert-base-multilingual-uncased`              | 96.23 | 96.06 | 96.07 | 96.15 | 96.09 | 96.12 ± 0.06
| `xlm-roberta-base`                            | 96.94 | 96.82 | 97.05 | 97.06 | 96.99 | 96.97 ± 0.09
| `electra-base-italian-xxl-cased` (1M)         | 97.27 | 97.39 | 97.23 | 97.22 | 97.23 | **97.27** ± 0.06

## NER

### EVALITA 2009

> In the Named Entity Recognition subtask, systems are required to recognize only the Named Entities occurring
> in a text; more specifically Person, Organization, Location and Geo-Political Entities (see the annotation
> report for more details). As in the previous edition of EVALITA, the evaluation will be based on the Italian
> Content Annotation Bank (I-CAB) where Named Entities are annotated in the IOB format (where "B-begin" and
> "I-inside" denote the tokens belonging to Named Entities and "O-outside" is used for all other tokens).
> The dataset that has been used for the NER task at EVALITA 2007 (525 news stories), will be distributed as
> development set, while the testset will consist of completely new data.

Unfortunately, some part of the EVALITA 2007 development set is part of the 2009 train set, so we don't report
dev scores here (they were above 99.9%).

Results (Test set)

| Model                                         | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| --------------------------------------------- | ----- | ----- | ----- | ----- | ----- | -------------- |
| `dbmdz/bert-base-italian-cased`               | 86.12 | 86.23 | 86.01 | 85.89 | 85.55 | 85.96 ± 0.23
| `dbmdz/bert-base-italian-uncased`             | 85.63 | 85.51 | 84.98 | 85.21 | 84.80 | 85.23 ± 0.31
| `dbmdz/bert-base-italian-xxl-cased`           | 88.33 | 88.29 | 88.45 | 88.24 | 88.27 | 88.32 ± 0.07
| `dbmdz/bert-base-italian-xxl-uncased`         | 87.62 | 88.20 | 88.40 | 88.52 | 87.76 | 88.10 ± 0.35
| `bert-base-multilingual-cased`                | 84.46 | 83.86 | 84.87 | 84.84 | 85.41 | 84.69 ± 0.51
| `bert-base-multilingual-uncased`              | 83.36 | 83.46 | 83.96 | 83.88 | 83.71 | 83.67 ± 0.23
| `xlm-roberta-base`                            | 84.23 | 84.83 | 84.41 | 84.51 | 83.81 | 84.36 ± 0.34
| `electra-base-italian-xxl-cased` (1M)         | 87.50 | 88.10 | 88.56 | 88.22 | 88.31 | **88.14** ± 0.35

# Model usage

All trained models can be used from the [DBMDZ](https://github.com/dbmdz) Hugging Face [model hub page](https://huggingface.co/dbmdz)
using their model name. The following models are available:

* Cased and uncased BERT models (medium size corpus): `dbmdz/bert-base-italian-cased` and `dbmdz/bert-base-italian-uncased`
* Cased and uncased BERT models (XXL size corpus): `dbmdz/bert-base-italian-xxl-cased` and `dbmdz/bert-base-italian-xxl-cased`
* *ELECTRA* model (discriminator and generator on XXL size corpus): `dbmdz/electra-base-italian-xxl-cased-discriminator` and `dbmdz/electra-base-italian-xxl-cased-generator`

Example usage with 🤗/Transformers:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "dbmdz/bert-base-italian-xxl-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)
```

This loads the Italian XXL BERT cased model. The recently introduced ELEC**TR**A base model can be loaded with:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "dbmdz/electra-base-italian-xxl-cased-discriminator"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelWithLMHead.from_pretrained(model_name)
```

# License

All models are licensed under [MIT](LICENSE).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our BERT models just open an issue
[in the DBMDZ BERT repo](https://github.com/dbmdz/berts/issues/new) or in
[this repo](https://github.com/stefan-it/italian-bertelectra/issues/new) 🤗

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
