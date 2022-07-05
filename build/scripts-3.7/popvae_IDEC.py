"""
Implementation for Improved Deep Embedded Clustering as described in paper:

        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.

Usage:
    Weights of Pretrained autoencoder for mnist are in './ae_weights/mnist_ae_weights.h5':
        python IDEC.py mnist --ae_weights ./ae_weights/mnist_ae_weights.h5
    for USPS and REUTERSIDF10K datasets
        python IDEC.py usps --update_interval 30 --ae_weights ./ae_weights/usps_ae_weights.h5
        python IDEC.py reutersidf10k --n_clusters 4 --update_interval 3 --ae_weights ./ae_weights/reutersidf10k_ae_weights.h5

Author:
    Xifeng Guo. 2017.4.30
"""


from time import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn import metrics



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def autoencoder_vecchio(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = Dense(dims[0], name='decoder_0')(h)
    
    model = Model(inputs=x, outputs=h)
    #loss = keras.losses.mean_squared_error(x,h)
    #model.add_loss(loss)

    return model




def autoencoder(dims):
    
    latent_dim = dims[-1]
        
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                      mean=0., stddev=0.5,seed=None)
        return z_mean + K.exp(z_log_var) * epsilon 
   
    #encoder
    input_seq = keras.Input(shape=(dims[0],))
    x=layers.Dense(500,activation="relu",name='e_0')(input_seq)
    x=layers.Dense(500,activation="relu",name='e_1')(x)
    x=layers.Dense(2000,activation="relu",name='e_2')(x)
    z_mean=layers.Dense(latent_dim,name='mean')(x)
    z_log_var=layers.Dense(latent_dim,name='std')(x)
    z = layers.Lambda(sampling,output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder=Model(input_seq,[z_mean,z_log_var,z],name='encoder')
    
    #decoder
    decoder_input=layers.Input(shape=(latent_dim,),name='z_sampling')
    x=layers.Dense(2000,activation="relu",name='d_2')(decoder_input)#was elu
    x=layers.Dense(500,activation="relu",name='d_1')(x)
    x=layers.Dense(500,activation="relu",name='d_0')(x)
    output=layers.Dense(dims[0],activation="sigmoid")(x) #hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
    decoder=Model(decoder_input,output,name='decoder')
    
    #end-to-end vae
    output_seq = decoder(encoder(input_seq)[2])
    vae = Model(input_seq, output_seq, name='vae')
    
    reconstruction_loss = keras.losses.mean_squared_error(input_seq,output_seq)
    reconstruction_loss *= dims[0]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    #kl_loss *= 5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return vae

    
    


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




class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)

   
    
    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')
        else:
            print('ae_weights must be given. E.g.')
            print('    python IDEC.py mnist --ae_weights weights.h5')
            exit()

        hidden = self.autoencoder.get_layer(name='encoder').output
        # prepare IDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden[2])
        self.model = Model(inputs=self.autoencoder.input,outputs=[clustering_layer, self.autoencoder.output])       
        
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        z_mean = self.model.get_layer('encoder').output[0]
        z_log_var = self.model.get_layer('encoder').output[1]
        
        
        #input_seq = self.autoencoder.input
        #output_seq = self.autoencoder.output
        def loss_modificata(z_mean,z_log_var):
            def loss_autoencoder(true, pred):
                # Reconstruction loss
                reconstruction_loss = keras.losses.mean_squared_error(true,pred)
                # KL divergence loss
                #kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
                #kl_loss = K.sum(kl_loss, axis=-1)
                #kl_loss *= -0.5
                #vae_loss = K.mean(reconstruction_loss + kl_loss)
                return reconstruction_loss#vae_loss 
            return loss_autoencoder
        
        self.model.compile(loss={'clustering':'kld','decoder':loss_modificata(z_mean, z_log_var)},#vae_loss(z_mean, z_log_var)},#loss_autoencoder},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer,
                           run_eagerly=True)
        
    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input, self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        return encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_dir='./results/idec'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x)[2])
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/idec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            print('ite: ',ite)
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break
            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=[p[index * self.batch_size::], 
                                                    x[index * self.batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                    x[index * self.batch_size:(index + 1) * self.batch_size]])
                index += 1
            # save intermediate model
            #if ite % save_interval == 0:
                # save IDEC model checkpoints
            print('saving model to:', save_dir + '/IDEC_model_' + str(ite) + '.h5')
            self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')
        
        return y_pred

#%%
if __name__ == "__main__":
    # setting the hyper parameters     
    
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'usps', 'reutersidf10k'])
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--maxiter', default=12e4, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=100, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default='euromds_weights.hdf5', help='This argument must be given')
    parser.add_argument('--save_dir', default='results/idec_var')
    args = parser.parse_args()
    print(args)

    # load dataset
    optimizer = SGD(lr=0.1, momentum=0.99)
    import json
    x = json.load(open('data/euromds/euromds.json','r'))
    x = np.array(x)
    #if exclude_data_duplicates == True:
        # exclude duplicate rows:
    x = np.unique(x,axis=0)

    # prepare the IDEC model
    idec = IDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=args.n_clusters, batch_size=args.batch_size)
    idec.initialize_model(ae_weights=args.ae_weights, gamma=args.gamma, optimizer=optimizer)
    #plot_model(idec.model, to_file='idec_model.png', show_shapes=True)
    idec.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time()
    y_pred = idec.clustering(x, y=None, tol=args.tol, maxiter=3,#args.maxiter,
                                 update_interval=args.update_interval, save_dir=args.save_dir)
   # print('acc:', cluster_acc(y, y_pred))
    print('clustering time: ', (time() - t0))
    
    
    
    
   