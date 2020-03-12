
import numpy as np
import math

def hydrophobic_moment(sequence,scale='Normalized_consensus',angle=100,is_in_degrees=True,normalize=True):
    # Input: peptide sequence
    # Output: hydrophobic moment.
    # If normalize=True (default): outputs hydrophobic moment normalized to peptide length
    # Normalized consensus scale should be fine
    # Angle should be 100 for alpha helix, 180 for beta sheet (I haven't found any evidence of selection for increased beta sheet hydrophobic mooment); 140 as a negative control
    scales={'Eisenberg':{'A':  0.25, 'R': -1.80, 'N': -0.64,'D': -0.72, 'C':  0.04, 'Q': -0.69,'E': -0.62, 'G':  0.16, 'H': -0.40,'I':  0.73, 'L':  0.53, 'K': -1.10,'M':  0.26, 'F':  0.61, 'P': -0.07,'S': -0.26, 'T': -0.18, 'W':  0.37,'Y':  0.02, 'V':  0.54},
'Normalized_consensus':{'A':0.62,'C':0.29,'D':-0.9,'E':-0.74,'F':1.19,'G':0.48,'H':-0.4,'I':1.38,'K':-1.5,'L':1.06,'M':0.64,'N':-0.78,'P':0.12,'Q':-0.85,'R':-2.53,'S':-0.18,'T':-.05,'V':1.08,'W':0.81,'Y':0.26}}
    hscale=scales[scale]
    sin_sum = 0
    cos_sum = 0
    moment=0
    for i in range(len(sequence)):
        hp=hscale[sequence[i]]
        angle_in_radians=i*angle
        if is_in_degrees:
            angle_in_radians = (i*angle)*math.pi/180.0
        sin_sum += hp*math.sin(angle_in_radians)
        cos_sum += hp*math.cos(angle_in_radians)
    moment = math.sqrt(sin_sum**2+cos_sum**2)
    if normalize:
        moment = moment/len(sequence)
    return moment

def hmoment_percentile(sequence,angle=100,nsamples=1000):
    # Gives the hydrophobic moment percentile for a sequence
    # That is, compares a peptide to shuffled versions of itself
    # Because random peptides of different amino acid compositions give different hydrophobic moments, this is a better metric for isolating AMP-like sequences than raw hydrophobic moment
    # Higher percentile means high hydrophobic moment
    unshuffled_h_moment = hydrophobic_moment(sequence,angle=angle)
    shuffled = [i for i in range(len(sequence))]
    percentile=0.0
    for i in range(nsamples):
        np.random.shuffle(shuffled)
        shuffled_seq=[sequence[j] for j in shuffled]
        shuffled_h_moment = hydrophobic_moment(shuffled_seq,angle=angle)
        if shuffled_h_moment < unshuffled_h_moment:
            percentile += 1.0/nsamples
    return percentile

print(hydrophobic_moment('WRRWWKILKAALAKLAK'))
print(hydrophobic_moment('RRWKWRKLAKVLTTLLRGGKRIQRL'))
print(hmoment_percentile('WRRWWKILKAALAKLAK'))
print(hmoment_percentile('RRWKWRKLAKVLTTLLRGGKRIQRL'))