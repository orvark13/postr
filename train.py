import click
import flair, torch
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence, Corpus
from flair.datasets import ColumnCorpus
from time import time
from sys import exit
from os import path
import logging
import common as c

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


@click.command()
@click.option(
    "--name",
    help=f"Name of a new model to train on the given fold.",
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
    "--subtoken-pooling",
    type=click.Choice(
        ["first_last", "last", "first"],
        case_sensitive=False,
    ),
    default="first_last",
    help="Subtoken pooling strategy",
)
@click.option(
    "--layers",
    type=click.Choice(["-1", "-2", "-1,-2", "-1,-2,-3"]),
    help="Pre-trained layers to use for fine-tuning",
    default="-1",
)
@click.option(
    "--mini-batch-size",
    type=click.Choice(["8", "16", "32"]),
    help="Mini batch size",
    default="16",
)
@click.option(
    "--extra-epochs",
    help="Extra epochs to pre-train using training data from MIM-GOLD fold.",
    default=0,
)
@click.option(
    "--max-epochs",
    help="Maximum number of epochs to fine-tune for.",
    default=10,
)
@click.option(
    "--learning-rate",
    help="Initial learning rate for AdamW.",
    default=5.0e-5,
)
@click.option(
    "--seed",
    default=c.default_seed,
    help=f'Set seed to use.',
    show_default=True,
)
def train(
    seed: int,
    name: str,
    data: str,
    tag_column: int,
    fold: int,
    subtoken_pooling: str,
    layers: str,
    mini_batch_size: str,
    extra_epochs: int,
    max_epochs: int,
    learning_rate: float
):
    """Train a new model."""
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
        flair.device = torch.device("cuda")
        torch.cuda.empty_cache()

    c.make_deterministic(seed)

    fold_str = format(fold, "02")
    model_folder = f"./{name}/{fold_str}/"
    model_path = path.abspath(model_folder)
    if path.isfile(model_path):
        exit(f"Model folder already exists '{model_folder}'.")

    c.make_sure_fold_files_exist(data, fold)
    data_folder = path.abspath(data)

    start = time()

    embeddings = TransformerWordEmbeddings(
        model=c.pretrained_model,
        layers=layers,
        subtoken_pooling=subtoken_pooling,
        fine_tune=True,
        allow_long_sentences=True,
    )

    print("Finished loading pre-trained LM:", time() - start, "s.")

    columns = {0: "text", tag_column: "tag"}
    corpus: Corpus = ColumnCorpus(
        data_folder,
        columns,
        train_file=f"{fold_str}TM.tsv",
        test_file=f"{fold_str}PM.tsv",
    )

    if extra_epochs > 0:
        print(
            f"Starting to continue pre-training for {extra_epochs} epochs using training data from MIM-GOLD"
        )
        for i in range(extra_epochs):
            for sentence in corpus.train:
                embeddings.embed(sentence)
            print("Finished embedding epoch", i)

    logger.info(corpus)
    print("Corpus ready", time() - start, "s.")

    tag_dictionary = corpus.make_label_dictionary("tag")
    print(tag_dictionary)
    logger.info(tag_dictionary)

    print("Dictionary ready", time() - start, "s.")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="tag",
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )

    trainer = ModelTrainer(tagger, corpus)
    trainer.fine_tune(
        model_path,
        learning_rate=learning_rate,
        mini_batch_size=int(mini_batch_size),
        mini_batch_chunk_size=1,
        max_epochs=max_epochs,
        checkpoint=True,
        use_final_model_for_eval=False,
    )

    print("Finished training", time() - start, "s.")

    tagger.print_model_card()


if __name__ == "__main__":
    train()
