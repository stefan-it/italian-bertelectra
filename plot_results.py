import click

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

@click.command()
@click.option("--results", required=True, type=str, help="Should point to results.csv")
@click.option("--split", required=True, type=str, help="Should be dev or test")
@click.option("--metric", required=True, type=str, help="Should be a valid metric like Accuracy oder F1-Score")
@click.option("--title", required=True, type=str, help="Should be plot title like UD PoS Tagging (Development)")
@click.option("--output_filename", required=True, type=str, help="Should be like description.png")
def plot_results(results, split, metric, title, output_filename):
    runs = set()
    models = set()

    model_split_result_dict = {}


    with open(results, "rt") as f_p:
        for line in f_p:
            line = line.rstrip()

            model_name, run, current_split, result = line.split(";")
            
            if current_split != split:
                continue
            
            models.add(model_name)
            runs.add(run)
            
            if model_name not in model_split_result_dict:
                model_split_result_dict[model_name] = [result]
            else:
                model_split_result_dict[model_name].append(result)
            
    print(model_split_result_dict)

    runs = [i for i in range(1, len(runs) + 1)]
    values = []
    columns = []

    for i, _ in enumerate(runs):
        current_results = []
        for model_name, model_run_results in model_split_result_dict.items():
            current_results.append(float(model_run_results[i]))
            if model_name not in columns:
                columns.append(model_name)
        values.append(current_results)

    data = pd.DataFrame(values, runs, columns=columns)

    fig = sns.lineplot(data=data, palette="tab10",linewidth=2.5)
    fig.set(xlabel="run", ylabel=metric)
    fig.set(xticks=runs)
    fig.set_title(title)
    fig.figure.savefig(output_filename) # yeah, latest seaborn compatibility ;)

if __name__ == "__main__":
    plot_results()  # pylint: disable=no-value-for-parameter
