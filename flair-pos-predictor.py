import click
import sys

from typing import List

from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger

@click.command()
@click.option("--data_folder", required=True, type=str, help="Should point to ./data/UD-Italian-<name> ")
@click.option("--case", required=True, type=str, help="Should be cased or uncased")
@click.option("--model_name", required=True, type=str, help="Should be path to trained Flair model, ending with best-model.pt")
@click.option("--dataset", required=True, type=str, help="Should be dev or test")
def run_prediction(data_folder, case, model_name, dataset):
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
    print(corpus)

    # Corpus configuration
    tag_type = "pos"
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    tagger: SequenceTagger = SequenceTagger.load(model_name)

    number_tags = 0
    number_correct_tags = 0

    ds = corpus.test if dataset == "test" else corpus.dev

    print(len(ds))

    for sentence in ds:
        tokens = sentence.tokens
        gold_tags = [token.get_tag('pos').value for token in sentence.tokens]

        tagged_sentence = Sentence()
        tagged_sentence.tokens = tokens

        tagger.predict(tagged_sentence)

        predicted_tags = [token.get_tag('pos').value for token in tagged_sentence.tokens]
        
        assert len(tokens) == len(gold_tags)
        assert len(gold_tags) == len(predicted_tags)
        
        number_tags += len(predicted_tags)
        
        for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
            if gold_tag == predicted_tag:
                number_correct_tags += 1

    print(f'Accuracy: {number_correct_tags / number_tags}')

if __name__ == "__main__":
    run_prediction()  # pylint: disable=no-value-for-parameter
