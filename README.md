# Antimicrobial-Peptides
This repository contains:

-In `data`, notebooks for scraping, preprocessing, and cleaning data from several databases of AMP information.
    -And the data itself, as scraped in April 2018.
  
-In `MIC_Prediction.ipynb`, code for loading the data, doing some exploratory analysis, and then training deep learning models.
    -These models predict [MIC](https://en.wikipedia.org/wiki/Minimum_inhibitory_concentration) for a given peptide using the sequence of amino acids in the peptide.
    -Additional code in the notebook generates sequences and uses simulated annealing to find sequences with good (low) predicted MIC.
    -Two especially low-MIC-predicted sequences were experimentally verified to have low MIC in vitro.

-In `Hemolysis`, unfinished work pointing in the direction of predicting toxicity of a peptide from its amino acid sequence.
