# 🇮🇹 Italian BERTELECTRA models

In this repository we monitor all experiments for our trained [DBMDZ models](https://github.com/dbmdz/berts)
for Italian. It includes both BERT and ELECTRA models and we use 🤗 Transformers library to fine-tune models.

Made with 🤗 and ❤️ from Munich.

# Experiments

We use the awesome 🤗 Transformers library for all fine-tuning experiments.

Please star and watch [Transformers](https://github.com/huggingface/transformers) on GitHub!

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
| `dbmdz/bert-base-italian-xxl-cased`           | 98.52 | 98.63 | 98.76 | 98.70 | 98.63 | **98.65** ± 0.08
| `dbmdz/bert-base-italian-xxl-uncased`         | 98.38 | 98.33 | 98.35 | 98.41 | 98.30 | 98.35 ± 0.04
| `bert-base-multilingual-cased`                | 98.50 | 98.54 | 98.49 | 98.54 | 98.43 | 98.50 ± 0.04
| `bert-base-multilingual-uncased`              | 98.24 | 98.17 | 98.22 | 98.27 | 98.23 | 98.23 ± 0.03
| `xlm-roberta-base`                            | 98.63 | 98.61 | 98.66 | 98.60 | 98.62 | 98.62 ± 0.02

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
