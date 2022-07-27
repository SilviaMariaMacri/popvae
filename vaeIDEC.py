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
import json

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



def load_data(dataset):
    
    if dataset=='mnist':
        from tensorflow.keras.datasets import mnist
        (x, y), (x_test, y_test) = mnist.load_data()
        #x = np.concatenate((x_train, x_test))
        #y = np.concatenate((y_train, y_test))
        x = x.reshape((x.shape[0], -1))
        x = np.divide(x, 255.)
        print('MNIST samples', x.shape)
        return x, y       
    
    if dataset == 'euromds':    
        ### load euromds dataset
        x = json.load(open('data/euromds/euromds.json','r'))
        x = np.array(x)
        #if exclude_data_duplicates == True:
        #    x = np.unique(x,axis=0)
        y = None
        return x,y


'''
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
'''


class IDEC_PopVAE(object):

    def __init__(self,latent_dim,stddev_epsilon,seed,real_dim,n_clusters,out):
        self.latent_dim = latent_dim
        self.stddev_epsilon = stddev_epsilon
        self.seed = seed
        self.real_dim = real_dim
        self.n_clusters = n_clusters
        self.out = out
    
    def vae(self,loss):
    
        def sampling(args):
            z_mean ,z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                      mean=0., stddev=self.stddev_epsilon,seed=self.seed)
            return z_mean + K.exp(z_log_var) * epsilon
            
        #encoder
        self.input_seq = keras.Input(shape=(self.real_dim,))
        x=layers.Dense(500,activation="relu")(self.input_seq)
        x=layers.Dense(500,activation="relu")(x)
        x=layers.Dense(2000,activation="relu")(x)
        
        self.z_mean=layers.Dense(self.latent_dim)(x)
        self.z_log_var=layers.Dense(self.latent_dim)(x)
        #z = Sampling()([z_mean, z_log_var])#
        z=layers.Lambda(sampling,output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
        
        self.encoder=Model(self.input_seq,[self.z_mean,self.z_log_var,z],name='encoder')#,z
        
        #decoder
        decoder_input=layers.Input(shape=(self.latent_dim,),name='z_sampling')
        x=layers.Dense(2000,activation="relu")(decoder_input)#was elu
        x=layers.Dense(500,activation="relu")(x)
        x=layers.Dense(500,activation="relu")(x)
        output=layers.Dense(self.real_dim,activation="sigmoid")(x) #hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
        decoder=Model(decoder_input,output,name='decoder')
        
        #end-to-end vae
        self.output_seq = decoder(self.encoder(self.input_seq)[2])
        self.vae = Model(self.input_seq, self.output_seq, name='vae')
        
        if loss == 'binary_crossentropy':
            reconstruction_loss = keras.losses.binary_crossentropy(self.input_seq,self.output_seq)
        if loss == 'mse':
            reconstruction_loss = keras.losses.mean_squared_error(self.input_seq,self.output_seq)
        
        reconstruction_loss *= self.real_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        #kl_loss *= 5
        self.vae_loss = K.mean(reconstruction_loss + kl_loss)
        
        self.vae.add_loss(self.vae_loss)
        
        #return self.vae
        

      
    def idec(self,coeff_vae_loss,gamma):
        
        #self.vae.load_weights(out+'/ae_weights.hdf5')
        
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output[0])#[2])
        self.idec = Model(self.vae.input, [clustering_layer,self.vae.output], name='idec')
        
        def target_distribution(q):
            weight = q ** 2 / K.sum(q,0)
            return tf.transpose(tf.transpose(weight) / K.sum(weight,1))

        q, _ = self.idec.output
        p = target_distribution(q)
        clustering_loss = keras.losses.kl_divergence(p,q)
        total_loss = coeff_vae_loss*self.vae_loss + gamma*clustering_loss
        self.idec.add_loss(total_loss) 
        '''
        
        loss = y_true * log(y_true / y_pred)
        
        
          y_pred = tf.convert_to_tensor(y_pred)
          y_true = tf.cast(y_true, y_pred.dtype)
          y_true = backend.clip(y_true, backend.epsilon(), 1)
          y_pred = backend.clip(y_pred, backend.epsilon(), 1)
          return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)
        '''
        #return self.idec
    
    def training(self,x,y,phase,optimizer,patience,batch_size,prediction_freq,max_epochs):
        
        if phase == 'ae':
            model = self.vae
            model.compile(optimizer)
            
        if phase == 'idec':
            model = self.idec
            model.compile(optimizer)
            
            print('initialize clustering')
            features = model.get_layer('encoder')(x)[0] #calcola sulla media
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            kmeans.fit_predict(features)
            model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
               
        checkpointer=keras.callbacks.ModelCheckpoint(
                      filepath=out+"/"+phase+"_weights.hdf5",
                      verbose=1,
                      save_weights_only=True, #controlla
                      best_only_model=True,
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
                    pred.to_csv(out+"/"+phase+"_training_preds.txt",sep='\t',index=False,mode='w',header=True)
                else:
                    pred.to_csv(out+"/"+phase+"_training_preds.txt",sep='\t',index=False,mode='a',header=False)
       
        print_predictions=keras.callbacks.LambdaCallback(
                 on_epoch_end=lambda epoch,
                 logs:saveLDpos(encoder=self.encoder,
                                predgen=x,
                                samples=y,
                                batch_size=batch_size,
                                epoch=epoch,
                                frequency=prediction_freq))

        #training
        t1=time.time()
        history=model.fit(x=x,
                        y=None,
                        shuffle=True,
                        epochs=max_epochs,
                        callbacks=[checkpointer,earlystop,reducelr,print_predictions],
                        validation_data=(x,None),
                        batch_size=batch_size)
        
        t2=time.time()
        vaetime=t2-t1
        print("VAE run time: "+str(vaetime)+" seconds")
               
        #save training history
        h=pd.DataFrame(history.history)
        h.to_csv(out+"/"+phase+"_history.txt",sep="\t")
                
        #predict latent space coords for all samples from weights minimizing val loss
        model.load_weights(out+"/"+phase+"_weights.hdf5")
        pred=model.get_layer('encoder')(x) #returns [mean,sd,sample] for individual distributions in latent space
        p=pd.DataFrame()
        for i in range(len(pred[0][0])):
            p['mean'+str(i)] = pred[0][:,i].numpy()
            p['sd'+str(i)] = pred[1][:,i].numpy()
        if y is not None:
            p['sampleID']=y
        p.to_csv(out+'/'+phase+'_latent_coords.txt',sep='\t',index=False)
        
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
        fig.savefig(out+"/"+phase+"_history.pdf",bbox_inches='tight')
        
    def load_weights(self, weights_path,phase): 
        if phase == 'ae':
            self.vae.load_weights(weights_path)
        if phase == 'idec':
            self.idec.load_weights(weights_path)
    
    def extract_features(self, x, phase):  
        if phase == 'ae':
            features = self.vae.get_layer('encoder')(x)
        if phase == 'idec':
            features = self.idec.get_layer('encoder')(x)
        return features

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.idec.predict(x, verbose=0)
        #q,_ = self.idec(x)
        q = q.numpy()
        return q.argmax(1)
        

if __name__ == "__main__": 

    
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="prova1",help="path for saving output")
    parser.add_argument('--dataset',default='mnist',choices=['mnist','eurodms'])
    parser.add_argument('--n_clusters',default=10)
    parser.add_argument('--ae_weights',default='ae_weights.hdf5',help="if None pretraining phase is run")
    parser.add_argument('--ae_weights_dir',default='out/mnist/pretraining/mnist0',help='ae_weights directory if pretraining not necessary')
    parser.add_argument('--coeff_vae_loss',default=0)
    parser.add_argument('--gamma',default=1)
    parser.add_argument("--loss",default='binary_crossentropy',choices=['binary_crossentropy','mse'])
    parser.add_argument("--opt",default='adam',choices=['sgd','adam'])
    parser.add_argument("--stddev_epsilon",default=0.5,help="std per epsilon nella funzione sampling")
    parser.add_argument("--latent_dim",default=10,type=int,help="N latent dimensions to fit. default: 2")
    parser.add_argument("--max_epochs",default=1000,type=int,help="max training epochs. default=500")
    parser.add_argument("--patience",default=50,type=int,help="training patience. default=50")
    parser.add_argument("--batch_size",default=256,type=int,help="batch size. default=32")
    parser.add_argument("--seed",default=None,type=int,help="random seed. \default: None")
    parser.add_argument("--prediction_freq",default=5,type=int,help="print predictions during training every \--prediction_freq epochs. default: 10")
    parser.add_argument("--gpu_number",default='0',type=str)
    args=parser.parse_args()
    
    out = args.out
    dataset = args.dataset
    n_clusters = args.n_clusters
    ae_weights = args.ae_weights
    ae_weights_dir = args.ae_weights_dir
    if args.ae_weights_dir is None:
        ae_weights_dir = out
    coeff_vae_loss = float(args.coeff_vae_loss)
    gamma = float(args.gamma)
    loss = args.loss
    optimizer = args.opt
    stddev_epsilon = args.stddev_epsilon    
    latent_dim = args.latent_dim
    max_epochs=args.max_epochs
    patience=args.patience
    batch_size=args.batch_size
    seed=args.seed
    prediction_freq=args.prediction_freq
    gpu_number=args.gpu_number
    
       
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    with open(out+'/config.json', 'w') as file:
        json.dump(vars(args), file)
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number
    
    if not seed==None:
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
    
    # load dataset
    x,y=load_data(dataset)
    real_dim = x.shape[1]
    
    model = IDEC_PopVAE(latent_dim,stddev_epsilon,seed,real_dim,n_clusters,out)
    model.vae(loss)
   
    if args.ae_weights is None:
        print('pretraining')
        phase='ae'
        model.training(x,y,phase,optimizer,patience,batch_size,prediction_freq,max_epochs)
        ae_weights = 'ae_weights.hdf5'
        
    print('clustering')
    model.load_weights(ae_weights_dir+'/'+ae_weights,'ae')
    model.idec(coeff_vae_loss,gamma) 
    #%%
    phase = 'idec'
    model.training(x,y,phase,optimizer,patience,batch_size,prediction_freq,max_epochs)
   
