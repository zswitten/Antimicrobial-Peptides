### Antimicrobial-Peptides

# Data
[GRAMPA](https://github.com/zswitten/Antimicrobial-Peptides/blob/master/data/grampa.csv) is a database of peptides and their antimicrobial activity against various bacteria. The database contains the following key columns:
_bacterium_: the target bacterium.
_sequence_: the sequence of amino acids that make up the peptide.
_strain_: the strain of bacterium, when available.
_value_: the MIC of the peptide on the bacterium.
_database_: the database from which the row's information was scraped.

The database also contains the following columns:
_modifications_: modifications that have been applied to the sequence.
_unit_: the unit of measurement of MIC, always uM.
_url_source_: a link to the database page from which the row's information was scraped.
_is_modified_: a binary column stating whether or not the sequence was modified.
_has_unusual_modification_: a binary column stating whether or not the sequence was modified in any way other than by c-terminal amidation.
_has_cterminal_amidation_: a binary column stating whether or not the sequence was modified with c-terminal amidation.
_datasource_has_modifications_: a column stating whether the database for that row contained modification information. When this column is False, the sequence may have been modified irrespective of the value of `is_modified`.

# Analysis

* In `data`, notebooks for scraping, preprocessing, and cleaning data from several databases of AMP information.
    * And the data itself, as scraped in April 2018.
  
* In `MIC_Prediction.ipynb`, code for loading the data, doing some exploratory analysis, and then training deep learning models.
    * These models predict [MIC](https://en.wikipedia.org/wiki/Minimum_inhibitory_concentration) for a given peptide using the sequence of amino acids in the peptide.
    * Additional code in the notebook generates sequences and uses simulated annealing to find sequences with good (low) predicted MIC.
    * Two especially low-MIC-predicted sequences were experimentally verified to have low MIC in vitro.
    
* In `Model_comparison.ipynb`, code for comparing different regression models (different neural network architectures, ridge, and kNN regression).

* In `Hemolysis`, unfinished work pointing in the direction of predicting toxicity of a peptide from its amino acid sequence.
