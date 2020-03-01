import pandas as pd
import numpy as np
from utils import sequence_to_vector, CHARACTER_DICT
from Bio import SeqIO
import requests

class AMPDataset():
    def __init__(self, exclude_list=None):
        self.data = pd.read_csv('../data/AMP/grampa.csv')
        self.data = self.filter_data(self.data, exclude_list=exclude_list)
        self.x1 = self.data.sequence
        species_cat = self.data.bacterium.astype('category')
        self.category_to_species_dict = dict(enumerate(species_cat.cat.categories))
        self.x2 = species_cat.cat.codes.values
        self.y = self.data.value.values

    def filter_data(self, data, exclude_list = None):
        # data = data[data.value < 2.9]
        if exclude_list:
            data = data[~data.sequence.isin(exclude_list)]
        return data

class HemolysisDataset():
    def __init__(self,exclude_list=None):
        self.data = pd.read_csv('../data/Hemolysis/Cleaned_hemolytic_data.csv')
        self.data = self.data[['Sequence','log10_HC50', 'Units','Uncertainty']]


class CPPDataset():
    def __init__(self,include_d_residues=False):
        cpp_seqs = []
        with open('../data/CPP/Unmodified_peptides.fa','rU') as handle:
            for record in SeqIO.parse(handle,'fasta'):
                cpp_seqs.append(record.seq)
        with open('../data/CPP/Modified_peptides.fa','rU') as handle:
            for record in SeqIO.parse(handle,'fasta'):
                cpp_seqs.append(record.seq)
        if include_d_residues:
            cpp_strs = [
                str(cpp_seq).upper() for cpp_seq in cpp_seqs
            ]
        else:
            cpp_strs = [
                str(cpp_seq) for cpp_seq in cpp_seqs
            ]
        cpp_strs = self.filter_cpps(cpp_strs)
        self.data = set(cpp_strs)

    def filter_cpps(self, cpp_strs):
        new_cpp_strs = []
        for cpp in cpp_strs:
            if all([c in CHARACTER_DICT for c in cpp]):
                new_cpp_strs.append(cpp)
        return new_cpp_strs

class UniprotDatasetHuman():
    def __init__(self):
        self.data = pd.read_csv('data/Uniprot/human_peptides.csv')
