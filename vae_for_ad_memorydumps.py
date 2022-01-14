'''
VAE/MLP starting-point based on: 
[1] Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." https://arxiv.org/abs/1312.6114

Evolved to an AD based on the algorithm presented in:
[2] An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection using reconstruction probability. Special Lecture on IE, 2(1), 1-18.

Representation loosely follows:
[3] Hou, S., Saas, A., Chen, L., & Ye, Y. (2016, October). Deep4maldroid: A deep learning framework for android malware detection based on linux kernel system call graphs. In 2016 IEEE/WIC/ACM International Conference on Web Intelligence Workshops (WIW) (pp. 104-111). IEEE.
& 
[4] Saxe, J., & Berlin, K. (2015, October). Deep neural network based malware detection using two dimensional binary program features. In 2015 10th International Conference on Malicious and Unwanted Software (MALWARE) (pp. 11-20). IEEE.

With further model insights taken from 

[5] Lopez-Martin, M., Carro, B., Sanchez-Esguevillas, A., & Lloret, J. (2017). Conditional variational autoencoder for prediction and feature recovery applied to intrusion detection in iot. Sensors, 17(9), 1967.

[6] Xu, H., Chen, W., Zhao, N., Li, Z., Bu, J., Li, Z., ... & Chen, J. (2018, April). Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications. In Proceedings of the 2018 World Wide Web Conference (pp. 187-196).

[7] https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html

[8] https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

[9] https://medium.com/@miguelmendez_/vaes-i-generating-images-with-tensorflow-f81b2f1c63b0

[10] https://jaan.io/what-is-variational-autoencoder-vae-tutorial/#mean-field

And with background in

[11] https://glassboxmedicine.com/2019/12/07/connections-log-likelihood-cross-entropy-kl-divergence-logistic-regression-and-neural-networks/

[12] https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-understanding-kl-divergence-2b382ca2b2a8

[13] https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

[14] http://colah.github.io/posts/2015-09-Visual-Information/

[15] https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29

[16] https://towardsdatascience.com/probability-concepts-explained-bayesian-inference-for-parameter-estimation-90e8930e5348

[17] https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1


'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import TensorBoard

import tensorflow_probability as tfp

import functools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import sys

import pandas as pd

'''
print("place quick tests here")
sys.exit(0)
'''

mseflag = False



#see Appendix C in [1]
def reconprob(args):

    x_true, x_decoded_mean, x_decoded_log_var = args
 
    normsum = K.sum(x_decoded_mean)
    scaled_x_decoded_mean=tf.map_fn(lambda x : x/normsum, x_decoded_mean)
    x_decoded_log_var = tf.add(x_decoded_log_var,1E-4) # As per [7]
    if mseflag: # mse is basically nll-gaussian with diagonal covariance 1. See [7]
        dist = tfp.distributions.MultivariateNormalDiag(loc=scaled_x_decoded_mean)
    else:
        dist = tfp.distributions.MultivariateNormalDiag(loc=scaled_x_decoded_mean, scale_diag=K.exp(0.5*x_decoded_log_var))
    reconprob = dist.prob(x_true)

    return reconprob 


def crossentropy_loss(pred_reconerr): # Tucking the loss function inside an output node is just simpler

    return pred_reconerr


#derivation exlainen in [7] Much clearer than using tfp.distributions.MultivariateNormalDiag.cross_entropy
def nll_gaussian(scaled_x_decoded_mean, x_decoded_log_var, x_true):

    ## element wise square
    square = tf.square(scaled_x_decoded_mean - x_true)## preserve the same shape as y_pred.shape
    ms = tf.add(tf.divide(square,K.exp(x_decoded_log_var)), K.log(K.exp(x_decoded_log_var)))
    ## axis = -1 means that we take mean across the last dimension 
    ## the output keeps all but the last dimension
    ## ms = tf.reduce_mean(ms,axis=-1)
    ## return scalar
    ms = tf.reduce_mean(ms)
    return(ms)

#see Appendix C in [1] + # Tucking the loss function inside an output node is just simpler
def reconerr(args):

    x_true, x_decoded_mean, x_decoded_log_var = args


    normsum = K.sum(x_decoded_mean)  
    scaled_x_decoded_mean = tf.map_fn(lambda x : x/normsum, x_decoded_mean) 
    x_decoded_log_var = tf.add(x_decoded_log_var,1E-4)

    if mseflag:   
        reconerr=mse(x_true, scaled_x_decoded_mean)
    else:
        reconerr=nll_gaussian(scaled_x_decoded_mean, x_decoded_log_var, x_true)

    # Assuming L=1
    return reconerr 


#Datase load + scale

def normalize(x, rowsum, histogram_dim):

    if rowsum > 0:
        return x/rowsum
    else:
        return 1/histogram_dim #Handles all-0 vector case

def rowratio(row): #histogram scaling to row ratio is insensitive "overall" to execution time + HANDLES probs with all-0 vectors
    histogram_dim = row.shape[0]    
    return row.apply(normalize, args=(row.sum(),histogram_dim)) 


#Datase load + scale
def get_dataset(file_path, **kwargs):

    df = pd.read_csv(file_path)
    x_train = df.loc[:, 'apk_AccessibilityManager':'apk_WindowManager']
    y_train = df.loc[:, 'apk_is_mal']
    labels = df.loc[:, 'apk_id':'apk_is_mal']

    x_train = x_train.astype(np.float64)
    x_train=x_train.apply(rowratio, axis=1)
    return x_train, y_train, labels




# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0] # see https://www.gsrikar.com/2017/06/what-is-tensor-rank-and-tensor-shape.html#:~:text=So%2C%20Rank%20is%20defined%20as,size%20of%20the%20each%20dimension.&text=A%20tensor%20with%20rank%201,are%20points%20on%20a%20line.
    dim = K.int_shape(z_mean)[1]
    # by default K.random_normal has mean = 0 and std = 1.0 see https://www.tensorflow.org/api_docs/python/tf/keras/backend/random_normal
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5* z_log_var) * epsilon  # (8) with L=1, .5 - as per [1]


def plot_ad_hprof_results(models, mse,
                 data,
                 batch_size,
                 model_name="vae_mlp_for_ad_hprof"):

    encoder, decoder = models
    x_test, y_test, labels = data
    os.makedirs(model_name, exist_ok=True)
    print("Predicting...")
    print(x_test.head())


    z_mean, z_log_var, _ = encoder.predict(x_test, batch_size=batch_size)
    rows, cols = z_mean.shape

    reconprob_scores = []

    for i in range(0,rows):
        print(str(i)+"/"+str(rows-1)+":")
        sys.stderr.write(str(i)+"/"+str(rows-1)+":\n")
        sys.stderr.flush()
        #print("test input: "+str(x_test.iloc[i]))
    
        if mse:
            z = K.random_normal(shape=(batch_size, cols), mean=z_mean[i] )
        else:
            z = K.random_normal(shape=(batch_size, cols), mean=z_mean[i], stddev=K.exp(0.5 * z_log_var[i]) )

        x_reconprob_list = []
        #for j in range(0,1): #Debug
        for j in range(0,batch_size):
            x_decoded_mean, x_decoded_log_var, x_reconerr, x_reconprob = decoder.predict([ tf.constant(x_test.iloc[i].to_numpy(), shape=[1,len(x_test.columns)]), tf.constant(z[j], shape=[1,cols]) ])
            x_reconprob_list.append(x_reconprob)
            #print("x_decoded_mean["+str(j)+"]: "+str(x_decoded_mean))
            #print("x_decoded_log_var["+str(j)+"]: "+str(x_decoded_log_var))
            #print("x_reconprob: "+str(x_reconprob))

        mean_x_reconprob = np.mean(x_reconprob_list)
        print("mean_x_reconprob: "+str(mean_x_reconprob))
        reconprob_scores.append(mean_x_reconprob)

    labels['ReconProb_Scores'] =  reconprob_scores
    labels.to_csv (model_name+'/'+model_name+'_scores.csv', index = False, header=True)

def vis_latent(models,
                 data,
                 batch_size=128,
                 model_name="vae_mlp_for_ad_hprof"):

    #Plots labels classification as a function of the 2D latent vector
    print("Plotting...")

    encoder, decoder = models
    x_test, y_test, labels = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_train)
    plt.colorbar(ticks=[0, 1])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    help_ = "Load h5 model trained weights"
    optional.add_argument("-w", "--weights", help=help_)
    help_ = "Path to train+val/test csv"
    required.add_argument('-f', '--fpath', help=help_, required=True)
    help_ = "Use mse loss instead of nll over gaussian (default)"
    optional.add_argument("-m", "--mse", help=help_, action='store_true')
    help_ = "Net topology: #1 - 50::25 #2 - 50:35::25  #3 - 50::25::2 "
    required.add_argument("-c", "--config", help=help_, required=True, type=int)

    args = parser.parse_args()

    #Load dataset - hprof
    x_train, y_train, labels = get_dataset(args.fpath)
    print(x_train.head())
    print(y_train.head())
    print(labels.head())
 

    topology = args.config

    # VAE model = encoder + decoder
    # build encoder model

    if topology == 1:

        #config1 - 50::25 # config 2 - 50:35::25  #config 3 - 50::25::2 
        # network parameters
        histogram_dim = 72
        input_shape = (histogram_dim, )
        intermediate_dim = 50 # Proportional to [2]
        batch_size = 128 # Must be > 100 for setting L=1
        latent_dim = 25 # As per [5] and propotional to [2]
        epochs = 2000 #8000 # ensuring convergence

        inputs = Input(shape=input_shape, name='encoder_input')
        ehidden = Dense(intermediate_dim, activation='relu', name='encoder_hidden')(inputs) #https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
        z_mean = Dense(latent_dim, name='z_mean')(ehidden)
        z_log_var = Dense(latent_dim, name='z_log_var')(ehidden)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var]) #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda?hl=en

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder') 
        encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_for_ad_hprof_encoder_config1.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        dhidden = Dense(intermediate_dim, activation='relu', name='decoder_hidden')(latent_inputs)
        x_mean = Dense(histogram_dim,  activation='sigmoid', name='x_mean')(dhidden) 
        x_log_var = Dense(histogram_dim, activation='sigmoid', name='x_log_var')(dhidden)
        x_recon_err = Lambda(reconerr, output_shape=(1,), name='x_recon_err')([inputs, x_mean, x_log_var])
        x_recon_prob = Lambda(reconprob, output_shape=(1,), name='x_recon_prob')([inputs, x_mean, x_log_var])

        # instantiate decoder model
        decoder = Model([inputs,latent_inputs],[x_mean, x_log_var, x_recon_err, x_recon_prob], name='decoder') 
        decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_for_ad_hprof_decoder_config1.png', show_shapes=True)

    elif topology == 2:

        #config1 - 50::25 # config 2 - 50:35::25  #config 3 - 50::25::2 
        # network parameters
        histogram_dim = 72
        input_shape = (histogram_dim, )
        intermediate_dim = 50 # Proportional to [2]
        intermediate_dim2 =  35 # Proportional to [2]
        batch_size = 128 # Must be > 100 for setting L=1
        latent_dim = 25 # 2D favours visualization
        epochs = 2000 #8000 # ensuring convergence

        inputs = Input(shape=input_shape, name='encoder_input')
        ehidden = Dense(intermediate_dim, activation='relu', name='encoder_hidden')(inputs) #https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
        ehidden2 = Dense(intermediate_dim2, activation='relu', name='encoder_hidden2')(ehidden)
        z_mean = Dense(latent_dim, name='z_mean')(ehidden2)
        z_log_var = Dense(latent_dim, name='z_log_var')(ehidden2)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var]) #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda?hl=en

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder') 
        encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_for_ad_hprof_encoder_config.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        dhidden2 = Dense(intermediate_dim2, activation='relu', name='decoder_hidden2')(latent_inputs)
        dhidden = Dense(intermediate_dim, activation='relu', name='decoder_hidden')(dhidden2)
        x_mean = Dense(histogram_dim,  activation='sigmoid', name='x_mean')(dhidden) 
        x_log_var = Dense(histogram_dim, activation='sigmoid', name='x_log_var')(dhidden)
        x_recon_err = Lambda(reconerr, output_shape=(1,), name='x_recon_err')([inputs, x_mean, x_log_var])
        x_recon_prob = Lambda(reconprob, output_shape=(1,), name='x_recon_prob')([inputs, x_mean, x_log_var])

        # instantiate decoder model
        decoder = Model([inputs,latent_inputs],[x_mean, x_log_var, x_recon_err, x_recon_prob], name='decoder') 
        decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_for_ad_hprof_decoder_config2.png', show_shapes=True)

    else:

        #config1 - 50::25 # config 2 - 50:35::25  #config 3 - 50::25::2 
        # network parameters
        histogram_dim = 72
        input_shape = (histogram_dim, )
        intermediate_dim = 50 # Proportional to [2]
        intermediate_dim2 = 25 # Proportional to [2]
        batch_size = 128 # Must be > 100 for setting L=1
        latent_dim = 2 # 2D favours visualization
        epochs = 2000 #8000 # ensuring convergence

        inputs = Input(shape=input_shape, name='encoder_input')
        ehidden = Dense(intermediate_dim, activation='relu', name='encoder_hidden')(inputs) #https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
        ehidden2 = Dense(intermediate_dim2, activation='relu', name='encoder_hidden2')(ehidden)
        z_mean = Dense(latent_dim, name='z_mean')(ehidden2)
        z_log_var = Dense(latent_dim, name='z_log_var')(ehidden2)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var]) #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda?hl=en

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder') 
        encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_for_ad_hprof_encoder_config3.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        dhidden2 = Dense(intermediate_dim2, activation='relu', name='decoder_hidden2')(latent_inputs)
        dhidden = Dense(intermediate_dim, activation='relu', name='decoder_hidden')(dhidden2)
        x_mean = Dense(histogram_dim,  activation='sigmoid', name='x_mean')(dhidden) 
        x_log_var = Dense(histogram_dim, activation='sigmoid', name='x_log_var')(dhidden)
        x_recon_err = Lambda(reconerr, output_shape=(1,), name='x_recon_err')([inputs, x_mean, x_log_var])
        x_recon_prob = Lambda(reconprob, output_shape=(1,), name='x_recon_prob')([inputs, x_mean, x_log_var])

        # instantiate decoder model
        decoder = Model([inputs,latent_inputs],[x_mean, x_log_var, x_recon_err, x_recon_prob], name='decoder') 
        decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_for_ad_hprof_decoder_config3.png', show_shapes=True)

    # instantiate VAE model
    [x_mean, x_log_var, x_recon_err, x_recon_prob] = decoder([inputs,encoder(inputs)[2]]) 
    vae = Model(inputs, [x_mean, x_log_var, x_recon_prob], name='vae_mlp_for_ad_hprof')

    models = (encoder, decoder) # passed individually to plot results since prediction is two-staged: Model parameter computation + generate using decoder


    reconstruction_loss = crossentropy_loss(x_recon_err) #see [1] appendix C + [7]


    #reconstruction_loss *= histogram_dim # This is hyperparameter C as per https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73  Does not apply for prob output
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) # Derivation in Appendix B of [1]
    kl_loss = K.sum(kl_loss, axis=-1) # Derivation in Appendix B of [1]
    kl_loss *= -0.5 # Derivation in Appendix B of [1]
    vae_loss = K.mean(reconstruction_loss + kl_loss) # Eq (7) in [2]: -ve ELBO
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam') # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20an%20optimization%20algorithm,iterative%20based%20in%20training%20data.&text=The%20algorithm%20is%20called%20Adam.

    if args.mse:
        mseflag=True #MSE is nll-gaussian with covariance 1


    if args.weights:
        print("Using trained model..")
        vae.load_weights(args.weights) # supply a trained .h5 file
        x_test = x_train
        y_test = y_train
        data = (x_test, y_test, labels)                

        if args.mse:
            postfix = "_mse"
        else:
            postfix = "_nll"


        if(topology == 1):
            cpostfix = "_conf1"
        elif(topology == 2):
            cpostfix = "_conf2"
        else:
            cpostfix = "_conf3"

        plot_ad_hprof_results(models, args.mse, data, batch_size=batch_size, model_name="vae_mlp_for_ad_hprof"+postfix+cpostfix)
        if(topology == 3):
            print(topology)
            vis_latent(models, data, batch_size=batch_size, model_name="vae_mlp_for_ad_hprof"+postfix+cpostfix)


    else:
        # train the autoencoder
        print("Starting training phase...")
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.82,  #70/15/15 split equivalent
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) #to monitor launch tensorboard --logdir=/tmp/autoencoder
        if mseflag:
            os.makedirs('vae_mlp_for_ad_hprof_mse_conf'+str(topology), exist_ok=True)
            vae.save_weights('vae_mlp_for_ad_hprof_mse_conf'+str(topology)+'/vae_mlp_for_ad_hprof_mse_conf'+str(topology)+'.h5')
            plot_model(encoder, to_file='vae_mlp_for_ad_hprof_mse_conf'+str(topology)+'/vae_mlp_for_ad_hprof_encoder_config'+str(topology)+'.png', show_shapes=True)
            plot_model(decoder, to_file='vae_mlp_for_ad_hprof_mse_conf'+str(topology)+'/vae_mlp_for_ad_hprof_decoder_config'+str(topology)+'.png', show_shapes=True)
        else:
            os.makedirs('vae_mlp_for_ad_hprof_nll_conf'+str(topology), exist_ok=True)
            vae.save_weights('vae_mlp_for_ad_hprof_nll_conf'+str(topology)+'/vae_mlp_for_ad_hprof_nll_conf'+str(topology)+'.h5')
            plot_model(encoder, to_file='vae_mlp_for_ad_hprof_nll_conf'+str(topology)+'/vae_mlp_for_ad_hprof_encoder_config'+str(topology)+'.png', show_shapes=True)
            plot_model(decoder, to_file='vae_mlp_for_ad_hprof_nll_conf'+str(topology)+'/vae_mlp_for_ad_hprof_decoder_config'+str(topology)+'.png', show_shapes=True)



