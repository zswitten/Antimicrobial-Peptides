import numpy as np
import math
import matplotlib.pyplot as plt
from Bio.SubsMat import MatrixInfo as matlist
import keras
from keras.layers import Dense, Dropout, LSTM, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, ZeroPadding1D, SimpleRNN, Bidirectional
from scipy.stats import kendalltau

CHARACTER_DICT = set([
    u'A', u'C', u'E', u'D', u'G', u'F', u'I', u'H', u'K',
    u'M', u'L', u'N', u'Q', u'P', u'S', u'R', u'T', u'W',
    u'V', u'Y']
)
nchar = len(CHARACTER_DICT)
CHARACTER_TO_INDEX = {
    character: i
    for i, character in enumerate(CHARACTER_DICT)
}
INDEX_TO_CHARACTER = {
    CHARACTER_TO_INDEX[c]: c
    for c in CHARACTER_TO_INDEX
}
FONT_TO_USE = 'Arial'
MAX_SEQUENCE_LENGTH = 50

def hydrophobic_moment(sequence,scale='Normalized_consensus',angle=0,is_in_degrees=True,normalize=True):
    # Angle should be 100 for alpha helix, 180 for beta sheet
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

def hmoment_analysis(test_sequence,num_sims=1000,angles=[100,140,160,180]):
#     100 degrees is alpha helix, 160 degrees is beta sheet (?)
    hmoments=[0]*len(angles)
    percentiles=[0]*len(angles)
    for k in range(len(angles)):
        test_angle=angles[k]
        import pdb; pdb.set_trace()
        hmoments[k] = hydrophobic_moment(test_sequence,angle=test_angle)
        other_h_moments=[0]*num_sims
        shuffled=list(range(len(test_sequence)))
        perGreater=0
        for i in range(num_sims):
            np.random.shuffle(shuffled)
            shuffled_seq=[test_sequence[j] for j in shuffled]
            other_h_moments[i]=hydrophobic_moment(shuffled_seq,angle=test_angle)
            if other_h_moments[i]<hydrophobic_moment(test_sequence,angle=test_angle):
                perGreater+=.1
        percentiles[k]=perGreater
    return hmoments,percentiles

def shuffle_seq(sequence):
    shuffled=list(range(len(sequence)))
    np.random.shuffle(shuffled)
    return [sequence[j] for j in shuffled]

def fourier_transformed_sequence(sequence_vector,frequencies):
    complex_fourier_val = np.zeros((len(frequencies),len(sequence_vector[0])))
    amplitudes = np.zeros((len(frequencies),len(sequence_vector[0])))
    phases = np.zeros((len(frequencies),len(sequence_vector[0])))
    imaginary = np.zeros((len(frequencies),len(sequence_vector[0])))
    real = np.zeros((len(frequencies),len(sequence_vector[0])))
    power_spectrum = np.zeros((len(frequencies),len(sequence_vector[0])))
    counter=0
    for i,frequency in enumerate(frequencies):
        for k,aa_vector in enumerate(sequence_vector):
            for l, val in enumerate(aa_vector):
                complex_fourier_val[i][l] += val*np.exp(-2j*math.pi*frequency*k)
    for i,row in enumerate(complex_fourier_val):
        for k, val in enumerate(row):
            power_spectrum[i][k] = abs(val)**2
            real[i][k] = val.real
            imaginary[i][k] = val.imag
            amplitudes[i][k] = abs(val)
            phases[i][k] = np.angle(val)

    return amplitudes,phases,power_spectrum,complex_fourier_val,real,imaginary

def fourier_transformed_embeddings(sequence_vectors,frequencies = [0.02*i for i in range(3,22)],embed_type='amplitude_and_phase'):
    to_return = []
    for mat in sequence_vectors:
        amplitudes,phases,power_spectrum,complex_fourier_val,real,imaginary = fourier_transformed_sequence(mat,frequencies)
        if embed_type == 'amplitude_and_phase':
            to_return.append(np.concatenate((np.array(amplitudes),np.array(phases)),axis=0))
        elif embed_type == 'real_and_imaginary':
            o_return.append(np.concatenate((np.array(real),np.array(imaginary)),axis=0))
    return np.array(to_return)

def sequence_to_vector(sequence, embed_dict=None):
# Screw c-terminal amidation
    if embed_dict==None:
        default = np.zeros([MAX_SEQUENCE_LENGTH, len(CHARACTER_TO_INDEX)])
        for i, character in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
            default[i][CHARACTER_TO_INDEX[character]] = 1
        return default
    else:
        default = np.zeros([MAX_SEQUENCE_LENGTH,len(embed_dict['A'])])
        for i, character in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
            for k,val in enumerate(embed_dict[character]):
                default[i,k] = val
        return default

def vector_to_sequence(vector, embed_dict=None):
    sequence = ''
    for row in vector:
        if 1 in row:
            sequence += INDEX_TO_CHARACTER[np.where(row == 1)[0][0]]
    return sequence

def power_spectrum_analysis(sequence_vectors,frequencies = [0.02*i for i in range(3,22)],num_to_try=-1,show_fig=True,save_fig=False,fname='',fig_title='Power spectrum plot'):
    all_power_spectra=[]
    for vec in sequence_vectors[:num_to_try]:
        amplitudes,phases,power_spectrum,complex_fourier = fourier_transformed_sequence(vec,frequencies)
        all_power_spectra.append(power_spectrum)

    average_power_spectrum = np.zeros((len(frequencies),len(sequence_vectors[0][0])))
    print('shape: '+repr(average_power_spectrum.shape))
    for spectrum in all_power_spectra:
        average_power_spectrum += spectrum
    average_power_spectrum /= len(all_power_spectra)
    # plt.plot(frequencies,average_power_spectrum)
    # plt.show()
    avg_spectrum = np.mean(average_power_spectrum,axis=1)
    plt.close()
    plt.plot(frequencies,avg_spectrum)
    print (avg_spectrum)
    plt.title(fig_title+' with n='+repr(len(sequence_vectors)))
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig('Figures/'+fname+'.png',bbox_inches='tight',frameon=False)
        my_file=open('Data_files/'+fname+'.txt','w')
        for i,freq in enumerate(frequencies):
            my_file.write(repr(freq)+','+repr(avg_spectrum[i])+'\n')
        my_file.close()
    return average_power_spectrum

def alignment_matrix_embedding(matrix_dict=matlist.blosum62,min_corr=0.97):
    # Uses distance geometry to generate an amino acid embedding
    # Keeps adding dimensions until the correlation between actual pairwise distances (per Blosum62) and reconstituted pairwise distances is >= min_corr
    distance_matrix = np.zeros([len(CHARACTER_DICT),len(CHARACTER_DICT)])
    for i,char1 in enumerate(CHARACTER_DICT):
        for j,char2 in enumerate(CHARACTER_DICT):
            if (char1,char2) in matrix_dict.keys():
                distance_matrix[i,j] = (matrix_dict[(char1,char1)]+matrix_dict[(char2,char2)]-2*matrix_dict[(char1,char2)]+0.)
            else:
                distance_matrix[i,j] = (matrix_dict[(char1,char1)]+matrix_dict[(char2,char2)]-2*matrix_dict[(char2,char1)]+0.)
    G_matrix = np.zeros([nchar,nchar])
    for i in range(nchar):
        for j in range(nchar):
            G_matrix[i,i] += distance_matrix[i,j]**2/nchar
            for k in range(nchar):
                G_matrix[i,i] -= distance_matrix[j,k]**2/(2*(nchar**2))

    for i in range(nchar):
        for j in range(nchar):
            G_matrix[i,j] = (G_matrix[i,i]+G_matrix[j,j]-distance_matrix[i,j]**2)/2
    values,vectors = np.linalg.eigh(G_matrix)
    corr = 0
    n_dimensions = 0
    while corr<min_corr:
        n_dimensions += 1
        sqrt_lambda_matrix = np.zeros([n_dimensions,n_dimensions])
        for i in range(n_dimensions):
            sqrt_lambda_matrix[i,i] = np.sqrt(values[i-n_dimensions])
        u_matrix = vectors[:,nchar-n_dimensions:nchar]
        product = np.matmul(sqrt_lambda_matrix,np.transpose(u_matrix))
        embedding_matrix = np.zeros([n_dimensions,nchar])
        for i in range(nchar):
            for j in range(n_dimensions):
                embedding_matrix[j,i] = product[n_dimensions-1-j,i]
        embedding_matrix = np.transpose(embedding_matrix)
        reconst_dist_matrix = np.zeros([nchar,nchar])
        for i in range(nchar):
            for j in range(nchar):
                to_set = np.linalg.norm(embedding_matrix[i,:]-embedding_matrix[j,:])
                reconst_dist_matrix[i,j] = to_set
        real_distances=[]
        reconst=[]
        for i in range(nchar-1):
            for j in range(i+1,nchar):
                real_distances.append(distance_matrix[i,j])
                reconst.append(reconst_dist_matrix[i,j])
        corr = np.corrcoef(real_distances,reconst)[0,1]
    embedding_dict = {}
    for i,char in enumerate(CHARACTER_DICT):
        embedding_dict[char] = embedding_matrix[i].tolist()
    return embedding_dict

def embed_sequences(sequences,method='one-hot'):
    if method == 'one-hot':
        sequence_vectors=[]
        correct_seqs=[]
        for sequence in sequences:
            try:
                sequence_vectors.append(sequence_to_vector(sequence))
                correct_seqs.append(sequence)
            except:
                pass
                # print(sequence)
        print('Starded with: '+repr(len(sequences)))
        print ('Ended with: '+repr(len(sequence_vectors)))

    if method == 'alignment_matrix':
        embedding_dict=alignment_matrix_embedding()
        sequence_vectors = []
        correct_seqs = []
        for sequence in sequences:
            try:
                sequence_vectors.append(sequence_to_vector(sequence,embed_dict=embedding_dict))
                correct_seqs.append(sequence)
            except:
                pass
        print('Starded with: '+repr(len(sequences)))
        print ('Ended with: '+repr(len(sequence_vectors)))
    return sequence_vectors

def conv_model(embed_length=len(CHARACTER_DICT),kernelsize=5):
    model = keras.models.Sequential()
    model.add(ZeroPadding1D(kernelsize, input_shape = (MAX_SEQUENCE_LENGTH, embed_length)))
    model.add(Conv1D(64,kernel_size = kernelsize,strides = 1,activation = 'relu',))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    #model.add(Dropout(0.5))
    model.add(Conv1D(64, kernelsize, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def dense_model(embed_length=len(CHARACTER_DICT)):
    model = keras.models.Sequential()
    model.add(Dense(256,activation='relu'))
    # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def conv_model_learn_embeddings(embedding_dim=6):
    model=keras.models.Sequential()
    model.add(Embedding(len(CHARACTER_DICT)+1,embedding_dim,input_length=MAX_SEQUENCE_LENGTH))
    model.add(ZeroPadding1D(kernelsize, input_shape = (MAX_SEQUENCE_LENGTH, embed_length)))
    model.add(Conv1D(64,kernel_size = kernelsize,strides = 1,activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    #model.add(Dropout(0.5))
    model.add(Conv1D(64, kernelsize, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def evaluate_model(model,test_x,test_y):
    predicted = model.predict(test_x)
    predicted = [pred[0] for pred in predicted]
    plt.plot(predicted,test_y,'.')
    plt.show()
    print(len(predicted))
    print(len(test_y))
    to_return = 'RMSE,'+repr(np.sqrt(float(np.average([(predicted[i]-test_y[i])**2 for i in range(len(test_y))]))))
    to_return += ',Pearson Correlation,'+repr(float(np.corrcoef(predicted,test_y)[1,0]))
    tau,pval = kendalltau(predicted,test_y)
    to_return += ',Kendall tau Correlation,'+repr(tau)
    to_return += ',MSE,'+repr(float(np.average([(predicted[i]-test_y[i])**2 for i in range(len(test_y))])))
    print (to_return)
    return to_return

def fourier_plus_conv(conv_embed_length=len(CHARACTER_DICT),fourier_embed_length=len(CHARACTER_DICT)):
    conv = keras.models.Sequential()
    conv.add(ZeroPadding1D(kernelsize, input_shape = (MAX_SEQUENCE_LENGTH, conv_embed_length)))
    conv.add(Conv1D(64,kernel_size = kernelsize,strides = 1,activation = 'relu',))
    conv.add(MaxPooling1D(pool_size=2, strides=2))
    #model.add(Dropout(0.5))
    conv.add(Conv1D(64, kernelsize, activation='relu'))
    conv.add(MaxPooling1D(pool_size=2))
    conv.add(Flatten())
    conv.add(Dropout(0.5))
    conv.add(Dense(100, activation='relu'))

    fourier = keras.models.Sequential()
    fourie.add(Dense(256,activation='relu'))
    # model.add(BatchNormalization())
    fourier.add(Flatten())
    fourier.add(Dense(128,activation='relu'))
    fourier.add(Dropout(0.5))
    fourier.add(Dense(64,activation='relu'))
    fourier.add(Dense(64,activation='relu'))

    model = keras.models.Sequential()
    model.add(Merge([conv,fourier],mode='concat'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')








