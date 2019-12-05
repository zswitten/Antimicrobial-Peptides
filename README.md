# Antimicrobial-Peptides

Data and code for [Deep learning regression model for antimicrobial peptide design](https://www.biorxiv.org/content/10.1101/692681v1.full). This repository contains code for training a model to predict antimicrobial activity of peptides against various bacteria including E. coli and P. aeruginosa.

## Data
GRAMPA [(link to csv file)](https://github.com/zswitten/Antimicrobial-Peptides/blob/master/data/grampa.csv) is a database of peptides and their antimicrobial activity against various bacteria. The database contains the following key columns:
- _bacterium_: the target bacterium.
- _sequence_: the sequence of amino acids that make up the peptide.
_strain_: the strain of bacterium, when available.
- _value_: the MIC of the peptide on the bacterium.

The database also contains the following auxiliary columns:
- _database_: the database from which the row's information was scraped.
- _url_source_: a link to the database page from which the row's information was scraped.
- _modifications_: modifications that have been applied to the sequence.
- _unit_: the unit of measurement of MIC, always uM.
- _is_modified_: a binary column stating whether or not the sequence was modified.
- _has_unusual_modification_: a binary column stating whether or not the sequence was modified in any way other than by c-terminal amidation.
- _has_cterminal_amidation_: a binary column stating whether or not the sequence was modified with c-terminal amidation.
- _datasource_has_modifications_: a column stating whether the database for that row contained modification information. When this column is False, the sequence may have been modified irrespective of the value of `is_modified`.

## Training a model
To train a model for E. coli that has a 1:1 ratio of random negative examples and runs for 60 epochs: 

```
git clone git@github.com:zswitten/Antimicrobial-Peptides.git
cd Antimicrobial-Peptides
pip install -r requirements.txt
python src/train_model.py --negatives=1 --bacterium='E. coli' --epochs=60
```

[This notebook](https://github.com/zswitten/Antimicrobial-Peptides/blob/master/MIC_Prediction.ipynb) contains code for reproducing the figures in the paper.
