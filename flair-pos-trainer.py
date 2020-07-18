import click
import sys

from typing import List

from flair.datasets import ColumnCorpus
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

@click.command()
@click.option("--data_folder", required=True, type=str, help="Should point to ./data/UD-Italian-<name> ")
@click.option("--case", required=True, type=str, help="Should be cased or uncased")
@click.option("--model_name", required=True, type=str, help="Should be a valid HuggingFace model hub name")
@click.option("--run_id", required=True, type=str, help="Should be [1-5]")
def run_experiment(data_folder, case, model_name, run_id):
    # Configuration
    column_format = {0: "token", 1: "pos"}
    train_file = f"train-{case}.txt"
    dev_file = f"dev-{case}.txt"
    test_file = f"test-{case}.txt"

    # Corpus
    corpus = ColumnCorpus(data_folder=data_folder,
                          column_format=column_format,
                          train_file=train_file,
                          dev_file=dev_file,
                          test_file=test_file)

    # Corpus configuration
    tag_type = "pos"
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # Embeddings
    embedding_types: List[TokenEmbeddings] = [
        TransformerWordEmbeddings(model=model_name, layers="all", use_scalar_mix=True)
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    # Trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # From base_dir (./data/UD_Italian-ISDT) take last "folder"
    experiment_name = data_folder.split("/")[-1]

    trainer.train(
        f"resources/taggers/pos-{experiment_name}-{model_name}-{run_id}",
        learning_rate=0.1,
        mini_batch_size=16,
        max_epochs=200,
        shuffle=True,
    )

if __name__ == "__main__":
    run_experiment()  # pylint: disable=no-value-for-parameter
