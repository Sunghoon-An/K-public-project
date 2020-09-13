import tensorflow
import keras
from keras import optimizers
### when using radam, need to
## Tensorflow addons
## below is addons import
##############################################################################################
import tensorflow_addons as tfa
from tfa.activation import gelum mish, rrelu
from tfa.optimizers import RectifiedAdam
from tfa.layers import GroupNormalization, InstanceNormalization, LayerNormalization, WeightNormalization, 
from tfa.optimizers import LazyAdam, ConditionalGradient, RectifiedAdam
from tfa.losses import ContrastiveLoss, NpairsMultilabelLoss, SparsemaxLoss, TripletSemiHardLoss
from tfa.metrics import MultiLabelConfusionMatrix
##############################################################################################

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import class_weight

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Sequential, Activation, Input, Dense, BatchNormalization, Dropout, add, Lambda, LeakyReLU, GaussianNoise, LeakyReLU, GaussianDropout, Highway
from keras.models import Sequential, Model
from keras.constraints import max_norm
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam, Adagrad, Adadelta
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint, TensorBoard
from keras.initializers import glorot_uniform

from config import *

## if you use conditional gradient, use custom func
####################################################
def frobenius_norm(m):
    """This function is to calculate the frobenius norm of the matrix of all
    layer's weight.
  
    Args:
        m: is a list of weights param for each layers.
    """
    total_reduce_sum = 0
    for i in range(len(m)):
        total_reduce_sum = total_reduce_sum + tf.math.reduce_sum(m[i]**2)
    norm = total_reduce_sum**0.5
    return norm
def CG_get_weight_norm:
    CG_frobenius_norm_of_weight = []
    CG_get_weight_norm = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda batch, 
        logs: CG_frobenius_norm_of_weight.append(frobenius_norm(model.trainable_weights).numpy())
    )
    return CG_get_wegiht_norm
'''example
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    callbacks=[CG_get_weight_norm])
    '''
####################################################

def RWN_model(input_dim):
    ## make your model
    return model

def Semi_model(input_dim):
    ## make your model
    return model

def DNN_model(input_dim):
    ## make your model
    return model