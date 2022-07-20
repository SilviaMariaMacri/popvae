#import warnings
import tensorflow.keras as keras
import numpy as np 
import os 
import allel
import pandas as pd
import time
import random
import subprocess, re, argparse
from matplotlib import pyplot as plt
from tensorflow.keras import layers
#from keras.layers.core import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Layer,InputSpec,Dense,Input

from sklearn.cluster import KMeans,SpectralClustering


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y

def load_data(dataset):
    
    if dataset=='mnist':
        dc, samples = load_mnist()
    
    if dataset == 'euromds':    
        ### load euromds dataset
        dc = json.load(open(infile,'r'))
        dc = np.array(dc)
        if exclude_data_duplicates == True:
            # exclude duplicate rows:
            dc = np.unique(dc,axis=0)
        samples = np.arange(len(dc))

    return dc,samples



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


#load model
def sampling(args):
    z_mean, z_log_var, latent_dim, stddev_epsilon, seed = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=stddev_epsilon,seed=seed)
    return z_mean + K.exp(z_log_var) * epsilon


def idec_ae(real_dim,latent_dim,stddev_epsilon,seed,loss,ae_weights_dir,n_clusters,coeff_vae_loss,gamma):
    #encoder
    input_seq = keras.Input(shape=(real_dim,))
    x=layers.Dense(500,activation="relu")(input_seq)
    x=layers.Dense(500,activation="relu")(x)
    x=layers.Dense(2000,activation="relu")(x)
    
    z_mean=layers.Dense(latent_dim)(x)
    z_log_var=layers.Dense(latent_dim)(x)
    #z = Sampling()([z_mean, z_log_var])#
    z=layers.Lambda(sampling,output_shape=(latent_dim,), name='z')([z_mean, z_log_var, latent_dim, stddev_epsilon, seed])
    
    encoder=Model(input_seq,[z_mean,z_log_var,z],name='encoder')#,z
    
    #decoder
    decoder_input=layers.Input(shape=(latent_dim,),name='z_sampling')
    x=layers.Dense(2000,activation="relu")(decoder_input)#was elu
    x=layers.Dense(500,activation="relu")(x)
    x=layers.Dense(500,activation="relu")(x)
    output=layers.Dense(real_dim,activation="sigmoid")(x) #hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
    decoder=Model(decoder_input,output,name='decoder')
    
    #end-to-end vae
    output_seq = decoder(encoder(input_seq)[2])
    
    vae = Model(input_seq, output_seq, name='vae')
    
    #get loss as xent_loss+kl_loss
    
    if loss == 'binary_crossentropy':
        reconstruction_loss = keras.losses.binary_crossentropy(input_seq,output_seq)
    if loss == 'mse':
        reconstruction_loss = keras.losses.mean_squared_error(input_seq,output_seq)
    
    reconstruction_loss *= real_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    #kl_loss *= 5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    vae.load_weights(ae_weights_dir+'/euromds_weights.hdf5')
    
    
    #clusterong layer
    
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output[0])#[2])
    
    idec = Model(vae.input, [clustering_layer,vae.output], name='idec')
    
    
    def target_distribution(q): 
        weight = q ** 2 / K.sum(q,0)
        return tf.transpose(tf.transpose(weight) / K.sum(weight,1))
    
    #q, _ = idec.predict(dc, verbose=0) #PROBLEMA forse questo calcolo non viene aggiornato??
    p = target_distribution(idec.output[0])
    clustering_loss = keras.losses.kl_divergence(p,clustering_layer)
    
    #total_loss = coeff_vae_loss*vae_loss + gamma*clustering_loss
    total_loss = K.mean(coeff_vae_loss*vae_loss+gamma*clustering_loss)
    #idec.add_loss(coeff_vae_loss*vae_loss)
    #idec.add_loss(gamma*clustering_loss)
    idec.add_loss(total_loss)
    
    return idec,encoder




if __name__ == "__main__":


    
    parser=argparse.ArgumentParser()
    parser.add_argument('--n_clusters',default=10)
    parser.add_argument('--dataset',default='mnist',choices=['mnist','eurodms'])
    parser.add_argument('--ae_weights_dir', default='results/server/mnist1', help='This argument must be given')
    parser.add_argument('--gamma',default=0.1)
    parser.add_argument('--coeff_vae_loss',default=1)
    parser.add_argument("--loss",default='binary_crossentropy',choices=['binary_crossentropy','mse'])
    parser.add_argument("--opt",default='adam',choices=['sgd','adam'])
    parser.add_argument("--exclude_data_duplicates",default=False)
    parser.add_argument("--stddev_epsilon",default=0.5,help="std per epsilon nella funzione sampling")
    parser.add_argument("--infile",default='data/euromds/euromds.json',
                        help="path to input genotypes in vcf (.vcf | .vcf.gz), \
                              zarr, or .popvae.hdf5 format. Zarr files should be as produced \
                              by scikit-allel's `vcf_to_zarr( )` function. `.popvae.hdf5`\
                              files store filtered genotypes from previous runs (i.e. \
                              from --save_allele_counts).")
    parser.add_argument("--out",default="prova",
                        help="path for saving output")
    parser.add_argument("--patience",default=50,type=int,
                        help="training patience. default=50")
    parser.add_argument("--max_epochs",default=1000,type=int,
                        help="max training epochs. default=500")
    parser.add_argument("--batch_size",default=32,type=int,
                        help="batch size. default=32")
    parser.add_argument("--save_allele_counts",default=False,action="store_true",
                        help="save allele counts and and sample IDs to \
                        out+'.popvae.hdf5'.")
    parser.add_argument("--save_weights",default=True,action="store_true",
                        help="save model weights to out+weights.hdf5.")
    parser.add_argument("--seed",default=None,type=int,help="random seed. \
                                                             default: None")
    parser.add_argument("--train_prop",default=0.9,type=float,
                        help="proportion of samples to use for training \
                              (vs validation). default: 0.9")
    parser.add_argument("--search_network_sizes",default=False,action="store_true",
                        help='run grid search over network sizes and use the network with \
                              minimum validation loss. default: False. ')
    parser.add_argument("--width_range",default="32,64,128,256,512",type=str,
                        help='range of hidden layer widths to test when `--search_network_sizes` is called.\
                              Should be a comma-delimited list with no spaces. Default: 32,64,128,256,512')
    parser.add_argument("--depth_range",default="3,6,10,20",type=str,
                        help='range of network depths to test when `--search_network_sizes` is called.\
                              Should be a comma-delimited list with no spaces. Default: 4,6,8,10')
    parser.add_argument("--depth",default=2,type=int,
                        help='number of hidden layers. default=6.')
    parser.add_argument("--width",default=128,type=int,
                        help='nodes per hidden layer. default=128')
    parser.add_argument("--gpu_number",default='0',type=str,
                        help='gpu number to use for training (try `gpustat` to get GPU numbers).\
                              Use ` --gpu_number "" ` to run on CPU, and  \
                              ` --parallel --gpu_number 0,1,2,3` to split batches across 4 GPUs.\
                              default: 0')
    parser.add_argument("--prediction_freq",default=5,type=int,
                        help="print predictions during training every \
                              --prediction_freq epochs. default: 10")
    parser.add_argument("--max_SNPs",default=None,type=int,
                        help="If not None, randomly select --max_SNPs variants \
                              to run. default: None")
    parser.add_argument("--latent_dim",default=10,type=int,
                        help="N latent dimensions to fit. default: 2")
    parser.add_argument("--prune_LD",default=False,action="store_true",
                        help="Prune sites for linkage disequilibrium before fitting the model? \
                        See --prune_iter and --prune_size to adjust parameters. \
                        By default this will use a 50-SNP rolling window to drop \
                        one of each pair of sites with r**2 > 0.1 .")
    parser.add_argument("--prune_iter",default=1,type=int,
                        help="number of iterations to run LD thinning. default: 1")
    parser.add_argument("--prune_size",default=50,type=int,
                        help="size of windows for LD pruning. default: 50")
    parser.add_argument("--PCA",default=False,action="store_true",
                        help="Run PCA on the derived allele count matrix in scikit-allel.")
    parser.add_argument("--n_pc_axes",default=20,type=int,
                        help="Number of PC axes to save in output. default: 20")
    parser.add_argument("--PCA_scaler",default="Patterson",type=str,
                        help="How should allele counts be scaled prior to running the PCA?. \
                              Options: 'None' (mean-center the data but do not scale sites), \
                              'Patterson' (mean-center then apply the scaling described in Eq 3 of Patterson et al. 2006, Plos Gen)\
                              default: Patterson. See documentation of allel.pca for further information.")
    parser.add_argument("--plot",default=False,action="store_true",
                        help="generate an interactive scatterplot of the latent space. requires --metadata. Run python scripts/plotvae.py --h for customizations")
    parser.add_argument("--metadata",default=None,
                        help="path to tab-delimited metadata file with column 'sampleID'.")
    args=parser.parse_args()
    
    ae_weights_dir = args.ae_weights_dir
    loss = args.loss
    opt = args.opt
    exclude_data_duplicates=args.exclude_data_duplicates
    stddev_epsilon=args.stddev_epsilon    
    infile=args.infile
    save_allele_counts=args.save_allele_counts
    patience=args.patience
    batch_size=args.batch_size
    max_epochs=args.max_epochs
    seed=args.seed
    save_weights=args.save_weights
    train_prop=args.train_prop
    gpu_number=args.gpu_number
    out=args.out
    prediction_freq=args.prediction_freq
    max_SNPs=args.max_SNPs
    latent_dim=args.latent_dim
    prune_LD=args.prune_LD
    prune_iter=args.prune_iter
    prune_size=args.prune_size
    PCA=args.PCA
    PCA_scaler=args.PCA_scaler
    depth=args.depth
    width=args.width
    n_pc_axes=args.n_pc_axes
    search_network_sizes=args.search_network_sizes
    plot=args.plot
    metadata=args.metadata
    depth_range=args.depth_range
    depth_range=np.array([int(x) for x in re.split(",",depth_range)])
    width_range=args.width_range
    width_range=np.array([int(x) for x in re.split(",",width_range)])
    dataset=args.dataset
    n_clusters = args.n_clusters
    gamma = args.gamma
    #ae_weights = args.ae_weights
    coeff_vae_loss = args.coeff_vae_loss
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    import json
    with open(out+'/config.json', 'w') as file:
        json.dump(vars(args), file)
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number
    
    if not seed==None:
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
    
    
    dc,samples=load_data(dataset)
    # divido dataset in train e test
    '''
    ninds=dc.shape[0]
    train=np.random.choice(range(ninds),int(train_prop*ninds),replace=False)
    test=np.array([x for x in range(ninds) if x not in train])
    traingen=dc[train,:]
    testgen=dc[test,:]
    '''
    traingen = dc.copy()
    testgen = dc.copy()
    
    
    
    real_dim=traingen.shape[1]
    idec,encoder=idec_ae(real_dim,latent_dim,stddev_epsilon,seed,loss,ae_weights_dir,n_clusters,coeff_vae_loss,gamma)
    
    #initialization
    features = encoder.predict(dc)[2]
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)
    idec.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    
    
    
    idec.compile(#loss={'clustering':clustering_loss,'decoder':vae_loss},
                 #loss_weights=[gamma,1],
                 optimizer=opt)#'adam'
    
    
    
    
    #################################################################### Lambda
     
    
    #callbacks
    checkpointer=keras.callbacks.ModelCheckpoint(
                  filepath=out+"/euromds_weights.hdf5",
                  verbose=1,
                  save_best_only=True,
                  monitor="val_loss",
                  period=1)
    
    earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0,
                                            patience=patience)
    
    reducelr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                               patience=int(patience/4),
                                               verbose=1,
                                               mode='auto',
                                               min_delta=0,
                                               cooldown=0,
                                               min_lr=0)
    
    def saveLDpos(encoder,predgen,samples,batch_size,epoch,frequency):
        if(epoch%frequency==0):
            pred=encoder.predict(predgen,batch_size=batch_size)[0]
            pred=pd.DataFrame(pred)
            pred['sampleID']=samples
            pred['epoch']=epoch
            if(epoch==0):
                pred.to_csv(out+"/euromds_training_preds.txt",sep='\t',index=False,mode='w',header=True)
            else:
                pred.to_csv(out+"/euromds_training_preds.txt",sep='\t',index=False,mode='a',header=False)
    
    print_predictions=keras.callbacks.LambdaCallback(
             on_epoch_end=lambda epoch,
             logs:saveLDpos(encoder=encoder,
                            predgen=dc,
                            samples=samples,
                            batch_size=batch_size,
                            epoch=epoch,
                            frequency=prediction_freq))
    
    
    #training
    t1=time.time()
    history=idec.fit(x=traingen,
                    y=None,
                    shuffle=True,
                    epochs=max_epochs,
                    callbacks=[checkpointer,earlystop,reducelr,print_predictions],
                    validation_data=(testgen,None),
                    batch_size=batch_size)
    
    t2=time.time()
    vaetime=t2-t1
    print("VAE run time: "+str(vaetime)+" seconds")
    
    #save training history
    h=pd.DataFrame(history.history)
    h.to_csv(out+"/euromds_history.txt",sep="\t")
    
    #predict latent space coords for all samples from weights minimizing val loss
    idec.load_weights(out+"/euromds_weights.hdf5")
    pred=encoder.predict(dc,batch_size=batch_size) #returns [mean,sd,sample] for individual distributions in latent space
    p=pd.DataFrame()
    p['mean1']=pred[0][:,0]
    p['mean2']=pred[0][:,1]
    p['sd1']=pred[1][:,0]
    p['sd2']=pred[1][:,1]
    pred=p
    #pred.columns=['LD'+str(x+1) for x in range(len(pred.columns))]
    pred['sampleID']=samples
    pred.to_csv(out+'/euromds_latent_coords.txt',sep='\t',index=False)
    
    #if not save_weights:
    #    subprocess.check_output(['rm',out+"/euromds_weights.hdf5"])
    
    if PCA:
        pcdata=np.transpose(dc)
        t1=time.time()
        print("running PCA")
        pca=allel.pca(pcdata,scaler=PCA_scaler,n_components=n_pc_axes)
        pca=pd.DataFrame(pca[0])
        colnames=['PC'+str(x+1) for x in range(n_pc_axes)]
        pca.columns=colnames
        pca['sampleID']=samples
        pca.to_csv(out+"/euromds_pca.txt",index=False,sep="\t")
        t2=time.time()
        pcatime=t2-t1
        print("PCA run time: "+str(pcatime)+" seconds")
    
    ######### plots #########
    #training history
    #plt.switch_backend('agg')
    fig = plt.figure(figsize=(3,1.5),dpi=200)
    plt.rcParams.update({'font.size': 7})
    ax1=fig.add_axes([0,0,1,1])
    ax1.plot(history.history['val_loss'][3:],"--",color="black",lw=0.5,label="Validation Loss")
    ax1.plot(history.history['loss'][3:],"-",color="black",lw=0.5,label="Training Loss")
    ax1.set_xlabel("Epoch")
    #ax1.set_yscale('log')
    ax1.legend()
    fig.savefig(out+"/euromds_history.pdf",bbox_inches='tight')
    
    if PCA:
        timeout=np.array([vaetime,pcatime])
        np.savetxt(X=timeout,fname=out+"/euromds_runtimes.txt")
    
    if plot:
        subprocess.run("python scripts/plotvae.py --latent_coords "+out+'/euromds_latent_coords.txt'+' --metadata '+metadata,shell=True)
    
    
    
    
    
