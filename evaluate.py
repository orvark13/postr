import click
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from sys import exit
from os import path
import common as c


@click.command()
@click.option(
    "--model",
    default=c.default_model,
    help=f'The name of an existing model under "{c.default_model}".',
    required=True,
)
@click.option(
    "--fold",
    help=f"The fold to use.",
    required=True,
    callback=c.validate_fold,
)
@click.option(
    "--data",
    default=c.data_folder,
    help=f"The name of the folder containing the data files from MIM-GOLD.",
    show_default=True,
)
@click.option(
    "--tag-column",
    type=click.Choice(["1", "2"]),
    help="Tag column number in data files, starting from 0.",
    default="2",
    show_default=True,
    callback=c.transform_tag_column,
)
@click.option(
    "--seed",
    default=c.default_seed,
    help=f'Set seed to use.',
    show_default=True,
)
def evaluate(seed, model, data, tag_column, fold):
    """Evaluate an existing model using a given a fold from MIM-GOLD."""

    c.make_deterministic(seed)

    fold_str = format(fold, "02")
    model_file = path.abspath(f"./{model}/{fold_str}/best-model.pt")
    if not path.isfile(model_file):
        model_file = path.abspath(f"./{model}/{fold_str}/final-model.pt")
    if not path.isfile(model_file):
        exit(
            f"Unable to find 'best-model.pt' or 'final-model.pt' in './{model}/{fold_str}/'."
        )
    print(f"Model: {model_file}")

    c.make_sure_fold_files_exist(data, fold)
    data_folder = path.abspath(data)

    # Assume that vocab.txt from pre-trained model can be found in path.
    tagger = SequenceTagger.load(model_file)

    columns = {0: "text", tag_column: "tag"}
    corpus = ColumnCorpus(
        data_folder,
        columns,
        train_file=f"{fold_str}TM.tsv",
        test_file=f"{fold_str}PM.tsv",
    )

    result = tagger.evaluate(
        corpus.test,
        gold_label_type="tag",
        out_path=f"./{model}/{fold_str}/delme_predictions.txt",
        allow_unk_predictions=True,
    )
    print(result.detailed_results)


if __name__ == "__main__":
    evaluate()
