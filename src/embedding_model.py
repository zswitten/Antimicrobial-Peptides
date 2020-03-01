import scipy
from Bio import SeqIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import sequence_to_vector, CHARACTER_DICT
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Dropout, LSTM, Conv2D,Input, Conv1D, MaxPooling1D, MaxPooling2D
from keras.layers import Flatten, ZeroPadding1D, SimpleRNN, Bidirectional, concatenate
from sklearn.decomposition import PCA
from sklearn import preprocessing
from make_dataset import AMPDataset, CPPDataset, HemolysisDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def conv_model_species_embedding(shape=[50,20],n_species=50,embed_dim=10):
	# Input should be [x0,x1] where the first x value is a sequence vector, and the second is an integer representing the bacterium_id
	# (See beginning of generate_pca_plot_from_data method for application of predictions)
	# NOTE: shuffling the data before training one of these models is VERY IMPORTANT (see the line x0,x1,y = shuffle(x0,x1,y) below)
	# If the data is not shuffled than the performance will be worse and the embeddings will be disrupted
    sequence_input = Input(
        shape=shape
    )
    zero_pad = ZeroPadding1D(5)(sequence_input)
    conv1 = Conv1D(
        64, kernel_size = 5, strides = 1,
        activation = 'relu'
    )(zero_pad)
    max_pool_1 = MaxPooling1D(
        pool_size = 2, strides = 2
    )(conv1)
    conv2 = Conv1D(
        64, kernel_size = 5,
        strides = 1, activation = 'relu'
    )(max_pool_1)
    max_pool_2 = MaxPooling1D(
        pool_size=2, strides=2
    )(conv2)
    flatten = Flatten()(max_pool_2)
    dropout = Dropout(0.5)(flatten)
    dense1 = Dense(
        100, activation='relu'
    )(dropout)
    embed_input = Input(shape=[1])
    embedding_layer=Embedding(n_species,embed_dim,input_length=1)
    bacteria_embedding = embedding_layer(embed_input)
    flatten_embed = Flatten()(bacteria_embedding)
    concat_embed = concatenate([dense1, flatten_embed])
    dense2_embed = Dense(50, activation='relu')(concat_embed)
    output_embed = Dense(1)(dense2_embed)
    model_embed = Model(inputs=[sequence_input, embed_input], outputs=output_embed)
    model_embed.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model_embed

def make_PCA_plot(species_embeddings,species_tags,colors):
	# Inputs: species embeddings (list of embeddings for the species you want to analyze)
	# Species_tags: species labels you want
	# Colors: list of colors to plot points (suggest coloring by gram negative/positive/fungi)
	to_embed = preprocessing.scale(species_embeddings)
	species_pca = PCA(n_components=2)
	species_pca.fit(to_embed)
	transformed = species_pca.fit_transform(to_embed)
	xvals = [val[0] for val in transformed]
	yvals = [val[1] for val in transformed]
	# PCA plot
	plt.scatter(xvals,yvals,color=colors)
	# plt.xrange()
	for i,point in enumerate(transformed):
	    plt.annotate(species_tags[i],(point[0]+0.1,point[1]-0.1),color=colors[i],fontsize=12)
	plt.savefig('../species_embeddings.png')
	plt.show()

def compare_multispecies_to_single_species_training(species,test_size=0.25,include_human=False):
	amp_data = get_amp_data_training_set(include_hemolysis=include_human)
	early_stopping = EarlyStopping(monitor = 'val_loss', patience = 15)

	amp_test_data = amp_data[0:int(len(amp_data)*test_size)]
	amp_train_data = amp_data[int(len(amp_data)*test_size):]
	amp_train_data_species_only = amp_train_data[amp_train_data.bacterium.isin([species])]
	amp_test_data = amp_test_data[amp_test_data.bacterium.isin([species])]
	amp_train_data = amp_train_data[~amp_train_data.sequence.isin(amp_test_data.sequence)]

	all_train_x0 = np.array([sequence_to_vector(s.upper()) for s in amp_train_data['sequence'].values])
	species_only_train_x0 = np.array([sequence_to_vector(s.upper()) for s in amp_train_data_species_only['sequence'].values])
	test_x0 = np.array([sequence_to_vector(s.upper()) for s in amp_test_data['sequence'].values])
	all_train_y = amp_train_data.value.values
	species_only_train_y = amp_train_data_species_only.value.values
	test_y = amp_test_data.value.values
	all_train_x2 = amp_train_data.reset_index()['bacterium_id']
	species_only_train_x2 = amp_train_data_species_only.reset_index()['bacterium_id']
	test_x2 = amp_test_data.reset_index()['bacterium_id']

	# Train model using species only
	species_only_embed_model = conv_model_species_embedding()
	species_only_embed_model.fit([species_only_train_x0,species_only_train_x2],species_only_train_y,epochs=40,validation_split=0.1,callbacks=[early_stopping])
	# Train model using embeddings & all data
	all_embed_model = conv_model_species_embedding()
	all_embed_model.fit([all_train_x0,all_train_x2],all_train_y,epochs=40,validation_split=0.1,callbacks=[early_stopping])

	# Evaluate
	print('Error from all species training: ' +repr(all_embed_model.evaluate([test_x0,test_x2],test_y)))
	print('Error from training on only one species: '+repr(species_only_embed_model.evaluate([test_x0,test_x2],test_y)))

	# Plot
	plt.plot(all_embed_model.predict([test_x0,test_x2]),test_y,'.')
	plt.savefig('All_data_for_training.png')
	plt.title('Error: '+repr(all_embed_model.evaluate([test_x0,test_x2],test_y)))
	plt.show()
	plt.plot(species_only_embed_model.predict([test_x0,test_x2]),test_y,'.')
	plt.savefig('Species_only_data_for_training.png')
	plt.title('Error: '+repr(species_only_embed_model.evaluate([test_x0,test_x2],test_y)))
	plt.show()

def get_amp_data_training_set(include_hemolysis=False):
	amp_data = AMPDataset().data
	amp_data = amp_data[[
	    'bacterium', 'modifications', 'is_modified', 'value', 'sequence', 'datasource_has_modifications',
	    'has_unusual_modification'
	]]
	amp_data = amp_data[amp_data.has_unusual_modification == False]
	amp_data = amp_data[amp_data['datasource_has_modifications'] == True]

	if include_hemolysis:
		hemolysis = HemolysisDataset()
		hemo_data = hemolysis.data
		hemo_data['bacterium'] = ['H. sapiens' for i in range(len(hemo_data))]
		hemo_data['value'] = hemo_data.log10_HC50
		hemo_data['sequence'] = [hemo_data.Sequence[i].upper() for i in range(len(hemo_data))]
		amp_data = pd.concat([amp_data,hemo_data],sort=False)
		keep_list = []
		for s in amp_data.sequence:
		    keep = True
		    for char in s:
		        if char not in CHARACTER_DICT:
		            keep = False
		    keep_list.append(keep)
		amp_data = amp_data[keep_list]

	amp_data = amp_data.groupby(['bacterium', 'sequence']).mean().reset_index()

	amp_data = amp_data[amp_data.bacterium.isin(amp_data.bacterium.value_counts().head(50).index)]
	amp_data.loc[:, 'bacterium_id'] = amp_data.bacterium.astype('category').cat.codes
	amp_data.loc[:,'bacterium_cat'] = amp_data.bacterium.astype('category')
	amp_data=amp_data.sample(frac=1).reset_index(drop=True)
	return amp_data


def generate_pca_plot_from_data(embed_dim=10,include_human=False):
	# Generates species embeddings based on the entire dataset (rather than a training test split, as this isn't really a training/test sort of thing)
	amp_data = get_amp_data_training_set(include_hemolysis = include_human)
	cat_dict = dict(enumerate(amp_data['bacterium_cat'].cat.categories))
	early_stopping = EarlyStopping(monitor = 'val_loss', patience = 15)
	bacterium_to_cat_id = {}
	for key in cat_dict:
	    bacterium_to_cat_id[cat_dict[key]]=key
	
	x0 = np.array([sequence_to_vector(s) for s in amp_data['sequence'].values])
	# Categorical input format for an embedding layer
	x1 = amp_data.reset_index()['bacterium_id']
	y = amp_data.value.values
	x0,x1,y = shuffle(x0,x1,y)

	
	model_embed = conv_model_species_embedding(shape=x0.shape[1:],n_species=len(amp_data.bacterium_id.value_counts()),embed_dim=embed_dim)

	# To this: using the entire dataset
	model_embed.fit([x0,x1],y,epochs=40,validation_split=0.1,callbacks=[early_stopping])

	embedding_layer = model_embed.layers[9]
	# Embeddings of top 10 species by amount of data, plus ESKAPE pathogens
	ec = embedding_layer.get_weights()[0][bacterium_to_cat_id['E. coli']]
	pa = embedding_layer.get_weights()[0][bacterium_to_cat_id['P. aeruginosa']]
	kp = embedding_layer.get_weights()[0][bacterium_to_cat_id['K. pneumoniae']]
	st = embedding_layer.get_weights()[0][bacterium_to_cat_id['S. typhimurium']]
	ab = embedding_layer.get_weights()[0][bacterium_to_cat_id['A. baumannii']]
	sm = embedding_layer.get_weights()[0][bacterium_to_cat_id['S. mutans']]
	sa = embedding_layer.get_weights()[0][bacterium_to_cat_id['S. aureus']]
	ml = embedding_layer.get_weights()[0][bacterium_to_cat_id['M. luteus']]
	bs = embedding_layer.get_weights()[0][bacterium_to_cat_id['B. subtilis']]
	efaecium = embedding_layer.get_weights()[0][bacterium_to_cat_id['E. faecium']]
	ca = embedding_layer.get_weights()[0][bacterium_to_cat_id['C. albicans']]
	sc = embedding_layer.get_weights()[0][bacterium_to_cat_id['S. cerevisiae']]
	efaecalis = embedding_layer.get_weights()[0][bacterium_to_cat_id['E. faecalis']]
	ecloacae = embedding_layer.get_weights()[0][bacterium_to_cat_id['E. cloacae']]
	species_tags = ['Ecoli','Pa','Kp','St','Ab','Sm','Sa','Ml','Bs','Efaecium','Ca','Sc','Efaecalis','Ecloacae']
	colors = ['b','b','b','b','b','g','g','g','g','g','k','k','g','b']
	to_embed = [ec,pa,kp,st,ab,sm,sa,ml,bs,efaecium,ca,sc,efaecalis,ecloacae]
	if include_human:
		human = embedding_layer.get_weights()[0][bacterium_to_cat_id['H. sapiens']]
		species_tags.append('Human')
		colors.append('r')
		to_embed.append(human)

	to_embed = preprocessing.scale(to_embed)
	make_PCA_plot(to_embed,species_tags,colors)

# generate_pca_plot_from_data(include_human=True)

compare_multispecies_to_single_species_training('H. sapiens',include_human=True)


