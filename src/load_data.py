import pandas as pd
import os
import ast
import numpy as np
import random
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Standardize units of MIC
def standardize_to_uM(concentration, unit, sequence):
    concentration = concentration.replace(' ', '')
    try:
        concentration = float(concentration)
    except:
        return None
    if unit == 'uM' or unit == u'\xb5M' or unit == u'uM)':
        return concentration
    elif unit == 'ug/ml' or unit == u'\xb5g/ml' or unit == u'ug/ml)':
        try:
            molWt = ProteinAnalysis(sequence).molecular_weight()
        except ValueError:
            return None
        return concentration * 1000/molWt
    elif unit == 'nmol/g' or unit == 'pmol/mg':
        #1g, at density of 1g/mL, is 1mL, so nmol/g is nmol/mL = umol/L = uM yay!
        return concentration
    else:
        # print 'Unit not recognized: ' + unit
        return None

# Enter an element of a result dictionary into df-ready row
def convert_result_to_rows(sequence, result):
    rows = []
    if 'bacteria' not in result:
        return rows
    for bacterium, strain in result['bacteria']:
        
        rows.append({
            'bacterium': bacterium,
            'strain': strain,
            'sequence': sequence.upper(),
            'url_source': result['url_sources'][0],
            'value': standardize_to_uM(
                result['bacteria'][(bacterium, strain)]['value'],
                result['bacteria'][(bacterium, strain)]['unit'],
                sequence
            ),
            'modifications': result['modifications'] if 'modifications' in result else [],
            'unit': 'uM'
        })
        if rows[-1]['value']:
            rows[-1]['value'] = np.log10(rows[-1]['value'])
    return rows

# Enter an element of a result dictionary into df-ready row
# Standardize units of MIC
def standardize_to_uM(concentration, unit, sequence):
    concentration = concentration.replace(' ', '')
    try:
        concentration = float(concentration)
    except:
        return None
    if unit == 'uM' or unit == u'\xb5M' or unit == u'uM)':
        return concentration
    elif unit == 'ug/ml' or unit == u'\xb5g/ml' or unit == u'ug/ml)':
        try:
            molWt = ProteinAnalysis(sequence).molecular_weight()
        except ValueError:
            return None
        return concentration * 1000/molWt
    elif unit == 'nmol/g' or unit == 'pmol/mg':
        #1g, at density of 1g/mL, is 1mL, so nmol/g is nmol/mL = umol/L = uM yay!
        return concentration
    else:
        # print 'Unit not recognized: ' + unit
        return None
    
def convert_result_to_rows(sequence, result):
    rows = []
    if 'bacteria' not in result:
        return rows
    for bacterium, strain in result['bacteria']:
        
        rows.append({
            'bacterium': bacterium,
            'strain': strain,
            'sequence': sequence.upper(),
            'url_source': result['url_sources'][0],
            'value': standardize_to_uM(
                result['bacteria'][(bacterium, strain)]['value'],
                result['bacteria'][(bacterium, strain)]['unit'],
                sequence
            ),
            'modifications': result['modifications'] if 'modifications' in result else [],
            'unit': 'uM'
        })
        if rows[-1]['value']:
            rows[-1]['value'] = np.log10(rows[-1]['value'])
    return rows

# Remove sequences with amino acids that aren't well-defined
def strip_sequences_with_char(df, bad_char):
    return df[~df.sequence.str.contains(bad_char)]

# We'll want to strip off any sequences with modifications that could be hard to replicate
# Their effects are too complex for the model
def is_modified(modifications_list):
    return len(modifications_list) > 0

# However, C-Terminal Amidation is common enough that we make an exception
CTERM_AMIDATION_TERMS = ['C-Terminal amidation','C-Terminus: AMD','C-Terminal','C-termianal amidation']

def has_non_cterminal_modification(modifications_list):
    return any(['C-Term' not in modification for modification in modifications_list])

def has_unusual_modification(modifications_list):
    return any([is_uncommon_modification(mod) for mod in modifications_list])

def has_cterminal_amidation(modifications_list):
    return any([is_cterminal_amidation(mod) for mod in modifications_list])

def has_disulfide_bonds(modifications_list):
    return any([is_disulfide_bond(mod) for mod in modifications_list])

def is_cterminal_amidation(mod):
    for term in CTERM_AMIDATION_TERMS:
        if term in mod:
            return True
    return False

def is_disulfide_bond(mod):
    return 'disulfide' in mod.lower()

def is_uncommon_modification(mod):
    return (not is_cterminal_amidation(mod)) and (not is_disulfide_bond(mod))

def datasource_has_modifications(cell, no_modification_data_sources):
    # Everything except CAMP and YADAMP has modification data
    return not any([s in cell for s in no_modification_data_sources])

def sequence_has_modification_data(cell, sequences_containing_modifications):
    # If the sequence is labeled modifictationless in another database it's OK
    return cell in sequences_containing_modifications

def correct_bacterium_typos(bacterium):
    typos = ['K. pneumonia','P. aeruginsa','S. aureu','S. a']
    corrections = ['K. pneumoniae','P. aeruginosa','S. aureus','S. aureus']
    typo2correction = zip(typos, corrections)
    for (typo, correction) in typo2correction:
        if bacterium == typo:
            return correction
        else:
            return bacterium

def clean_data(df):
    columns = df.columns

    for bad_char in ['U', 'X', 'Z', 'C']:
        df = strip_sequences_with_char(df, bad_char)

    df['is_modified'] = df.modifications.apply(is_modified)
    df['has_unusual_modification'] = df.modifications.apply(has_unusual_modification)
    df['has_cterminal_amidation'] = df.modifications.apply(has_cterminal_amidation)

    # Clean sequences by removing newlines and one improper sequence
    df.sequence = df.sequence.str.strip()
    df = df.loc[df.sequence != '/']

    # Exclude sequences with modifications
    # Exclude rows from YADAMP and CAMP for having no modification data
    #     Unless that sequence is in another DB

    df = df.loc[df.has_unusual_modification == False]

    no_modification_data_sources = ['camp3', 'yadamp']
    df['datasource_has_modifications'] = df['url_source'].apply(
        lambda x: datasource_has_modifications(x, no_modification_data_sources)
    )

    sequences_containing_modifications = set(
        df.loc[df.datasource_has_modifications == True,
        'sequence']
    )

    df['sequence_has_modifications'] = df['sequence'].apply(
        lambda x: sequence_has_modification_data(x, sequences_containing_modifications)
    )

    df['modification_verified'] = df['sequence_has_modifications'] | df['datasource_has_modifications']

    df = df.loc[df.modification_verified == True]

    # Correct typos, for example, 'P. aeruginsa'
    df['bacterium'] = df.bacterium.apply(correct_bacterium_typos)

    return df[columns]

def average_over_databases(df):
    return df.groupby('bacterium', 'sequence').mean().reset_index().dropna()

def load_df_from_dbs(data_path):
    # The scripts stored the outputs as dictionaries.
    all_results = []
    for f in os.listdir(data_path):
        if '.data' in f:
            with open(data_path + f, 'r') as g:
                all_results.append(ast.literal_eval(g.read()))
    # Load all the rows into an array
    rows = []
    for result_set in all_results:
        for sequence in result_set:
            for row in convert_result_to_rows(sequence, result_set[sequence]):
                rows.append(row)
    df = pd.DataFrame(rows)
    df = clean_data(df)
    return df