# Transformer-based POS tagger for Icelandic

This repository contains the source code for the Transformer POS tagger model created for my MSc. project.

The trained models that are the basis for the evaluation of the four model configuration reported in the project are also made available.

<img src="./_static/transformer_comparison.png" alt="Comparison of transformer models" width="500"/>

There are scripts to train and evaluate models new models. See subsections.

## Trained models

The trained models for the four model configurarions and 10-folds are provided. There are `.zip` files named A, B, C, and D that match models with same designations.

[Download models from Dropbox](https://www.dropbox.com/sh/3vfa3gbjypj3ang/AACq-ObhNhXLWph6Pe1RynjIa?dl=0)

## Training

```
python ./train.py --name x --fold 1
```

Use `--help` to see all parameter options.

## Evaluating

```
python ./evaluate.py --model x --fold 1
```

Use `--help` to see all parameter options.

## Data

[Version 21.05 of MIM-GOLD](https://clarin.is/en/resources/gold/) needs to be downloaded from CLARIN-IS. The scripts assume it is in a subfolder, but the parameter `--data` can be used to point to a folder with the fold sets.

```
wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/114/MIM-GOLD-SETS-21.05.zip
unzip -o "MIM-GOLD-SETS-21.05.zip"
```

The training script handles downloading [jonfd/convbert-base-igc-is](https://huggingface.co/jonfd/convbert-base-igc-is) from Hugging Face.
