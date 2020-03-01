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

def conv_model_species_embedding(shape=[50,20],n_species=50,embed_dim=10):
	# Input should be [x0,x1] where the first x value is a sequence vector, and the second is an integer representing the bacterium_id
	# (See beginning of generate_pca_plot_from_data method for application of predictions)
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


def generate_pca_plot_from_data(amp_data,embed_dim=10):
	# Generates species embeddings based on the entire dataset (rather than a training test split, as this isn't really a training/test sort of thing)
	amp_data = amp_data[[
	    'bacterium', 'modifications', 'is_modified', 'value', 'sequence', 'datasource_has_modifications',
	    'has_unusual_modification'
	]]
	amp_data = amp_data[amp_data.has_unusual_modification == False]
	amp_data = amp_data[amp_data['datasource_has_modifications'] == True]
	amp_data = amp_data.groupby(['bacterium', 'sequence']).mean().reset_index()

	amp_data = amp_data[amp_data.bacterium.isin(amp_data.bacterium.value_counts().head(50).index)]
	amp_data.loc[:, 'bacterium_id'] = amp_data.bacterium.astype('category').cat.codes
	amp_data.loc[:,'bacterium_cat'] = amp_data.bacterium.astype('category')
	cat_dict = dict(enumerate(amp_data['bacterium_cat'].cat.categories))
	early_stopping = EarlyStopping(monitor = 'val_loss', patience = 15)
	bacterium_to_cat_id = {}
	for key in cat_dict:
	    bacterium_to_cat_id[cat_dict[key]]=key
	
	x0 = np.array([sequence_to_vector(s) for s in amp_data['sequence'].values])
	# Categorical input format for an embedding layer
	x1 = amp_data.reset_index()['bacterium_id']
	y = amp_data.value.values
	model_embed = conv_model_species_embedding(shape=x0.shape[1:],n_species=len(amp_data.bacterium_id.value_counts()),embed_dim=embed_dim)
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
	to_embed = preprocessing.scale(to_embed)
	make_PCA_plot(to_embed,species_tags,colors)

generate_pca_plot_from_data(AMPDataset().data)



