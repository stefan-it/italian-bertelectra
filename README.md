# 🇮🇹 Italian BERTELECTRA models

In this repository we monitor all experiments with our trained [DBMDZ models](https://github.com/dbmdz/berts)
for Italian. It includes both BERT and ELECTRA models and we use Flair with a feature-based approach
for all evaluations.

Made with 🤗 and ❤️ from Munich.

# Experiments

We use the awesome Flair library for experiments with our Transformer-based models.
It nicely wraps the outstanding 🤗 Transformers library and we can also use the
HuggingFace model hub.

Please star and watch [Flair](https://github.com/flairNLP/flair) and [Transformers](https://github.com/huggingface/transformers)
on GitHub!

## PoS Tagging

### Italian-ISDT

Description:

> The Italian corpus annotated according to the UD annotation scheme was obtained by conversion
> from ISDT (Italian Stanford Dependency Treebank), released for the dependency parsing shared
> task of Evalita-2014 (Bosco et al. 2014).

Details:

* [Italian-ISDT Repository](https://github.com/UniversalDependencies/UD_Italian-ISDT)
* Commit: `f20fa2b`

Results:

| Model                                  | Run 1         | Run 2         | Run 3         | Run 4         | Run 5         | Avg.
| -------------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| `dbmdz/bert-base-italian-cased`        | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx |
| `dbmdz/bert-base-italian-uncased`      | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx |
| `dbmdz/bert-base-italian-xxl-cased`    | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx |
| `dbmdz/bert-base-italian-xxl-uncased`  | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx |
| `dbmdz/electra-base-italian-cased`     | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx |
| `dbmdz/electra-base-italian-xxl-cased` | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx | (xx.xx) xx.xx |