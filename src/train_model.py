from load_data import load_df_from_dbs
from nn import conv_model, evaluate, evaluate_as_classifier
from settings import MAX_SEQUENCE_LENGTH, character_to_index, CHARACTER_DICT, max_mic_buffer, MAX_MIC

from sklearn.model_selection import train_test_split
import numpy as np
import random
from Bio import SeqIO
import argparse

def get_bacterium_df(bacterium, df):
    bacterium_df = df.loc[(df.bacterium.str.contains(bacterium))].groupby(['sequence', 'bacterium'])
    return bacterium_df.mean().reset_index().dropna()


def sequence_to_vector(sequence):
    vector = np.zeros([MAX_SEQUENCE_LENGTH, len(character_to_index) + 1])
    for i, character in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
        vector[i][character_to_index[character]] = 1
    return vector

def generate_random_sequence(min_length=5, max_length=MAX_SEQUENCE_LENGTH, fixed_length=None):
    if fixed_length:
        length = fixed_length
    else:
        length = random.choice(range(min_length, max_length))
    sequence = [random.choice(list(CHARACTER_DICT)) for _ in range(length)]
    return sequence

def add_random_negative_examples(vectors, labels, negatives_ratio):
    if negatives_ratio == 0:
        return vectors, labels
    num_negative_vectors = int(negatives_ratio * len(vectors))
    negative_vectors = np.array(
        [sequence_to_vector(generate_random_sequence()) for _ in range(num_negative_vectors)]
    ) 
    vectors = np.concatenate((vectors, negative_vectors))
    negative_labels = np.full(num_negative_vectors, MAX_MIC)
    labels = np.concatenate((labels, negative_labels))
    return vectors, labels

def load_uniprot_negatives(count):
    uniprot_file = 'data/Fasta_files/Uniprot_negatives.txt'
    fasta = SeqIO.parse(uniprot_file, 'fasta')
    fasta_sequences = [str(f.seq) for f in fasta]
    negatives = []
    for seq in fasta_sequences:
        if 'C' in seq:
            continue
        start = random.randint(0,len(seq)-MAX_SEQUENCE_LENGTH)
        negatives.append(seq[start:(start+MAX_SEQUENCE_LENGTH)])
        if len(negatives) >= count:
            return negatives
    return negatives

def uniprot_precision(model):
    negatives = load_uniprot_negatives(1000)
    vectors = []
    for seq in negatives:
        try:
            vectors.append(sequence_to_vector(seq))
        except KeyError:
            continue
    preds = model.predict(np.array(vectors))
    false_positives = len([p for p in preds if p < MAX_MIC - max_mic_buffer])
    return 1 - false_positives / len(negatives)


def train_model(bacterium, negatives_ratio=1, epochs=100):
    """
    Bacterium can be E. coli, P. aeruginosa, etc.
    When with_negatives is False, classification error will be 0
    and error on correctly classified/active only/all will be equal
    because all peptides in the dataset are active
    """
    DATA_PATH = 'data/'
    df = load_df_from_dbs(DATA_PATH)
    bacterium_df = get_bacterium_df(bacterium, df)
    print("Found %s sequences for %s" % (len(bacterium_df), bacterium))
    bacterium_df['vector'] = bacterium_df.sequence.apply(sequence_to_vector)

    x = np.array(list(bacterium_df.vector.values))
    y = bacterium_df.value.values
    x, y = add_random_negative_examples(x, y, negatives_ratio)

    train_x, test_x, train_y, test_y = train_test_split(x, y)

    model = conv_model()
    model.fit(train_x, train_y, epochs=epochs)
    print("Avg. MIC error (correctly classified, active only, all)")
    print(evaluate(model, test_x, test_y))
    print('True positives, true negatives, false positives, false negatives')
    true_positives, true_negatives, false_positives, false_negatives = evaluate_as_classifier(
        model, test_x, test_y
    )
    print(true_positives, true_negatives, false_positives, false_negatives)
    print("Accuracy:",
        (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    )
    print("Precision on Uniprot:")
    print(uniprot_precision(model))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bacterium', type=str, default='E. coli', help='Name of bacterium, in single quotes')
    parser.add_argument('--negatives', type=float, default=1, help='Ratio of negatives to positives')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    train_model(bacterium=args.bacterium, negatives_ratio=args.negatives, epochs=args.epochs)
