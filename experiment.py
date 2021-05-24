# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# See the LICENSE file for more details.

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm


class DataSetLoader():
    def __init__(self, batch_size=32, size=(224,224), preprocess_image=(lambda x : x)):
        self.size = size
        self.batch_size = batch_size
        self.preprocess_image = preprocess_image
        
class TfDataSetLoader(DataSetLoader):
    
    def __init__(self, name, batch_size=32, size=(224,224), preprocess_image=(lambda x : x)):
        super().__init__(batch_size=batch_size, size=size, preprocess_image=preprocess_image)
        
        dataset, info = tfds.load(name=name, with_info=True)
        
        self.num_classes = info.features['label'].num_classes        
        self.label_dict = {i: info.features['label'].int2str(i) for i in range(self.num_classes)}
        
        if name == 'cats_vs_dogs':
            split=.8
            
            s = len(dataset['train'])
            s = int(s * split)
            
            d = dataset['train'].shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False)
            dataset = {
                'train': d.take(s),
                'test': d.skip(s)
            }
            
                
        self.dataset = {key: self.prepare_dataset(dataset[key], shuffle=(key=='train'))
            for key in ['train', 'test']} 
    
    def preprocess(self, data):
        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        image = tf.image.resize(image, self.size)
        image *= 255.
        image = self.preprocess_image(image)
        
        label = data['label']

        return image, label
    
    def prepare_dataset(self, dataset, shuffle):
        dataset = dataset.map(self.preprocess, num_parallel_calls=4)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        else:
            dataset = dataset.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1000)
        return dataset 
    
def global_softmax(i):
    o = tf.reshape(i, [-1, i.shape[1] * i.shape[2]])
    o = tf.nn.softmax(o)
    o = tf.reshape(o, [-1, i.shape[1],  i.shape[2]])
    
    return o

def gumbel(shape):
  return -(tf.math.log(-tf.math.log(tf.random.uniform(shape=shape))))

def make_x_block(input_shape, k=4, tau=.1, pre_tau=-1., threshold=None, auto_threshold_factor=.98,
            extra_layers_channels=[], name='x_block', sampling=False, activation='relu', fixed_k_sampling=False,
           concept_vector=False, sigmoid=False):
    i = tf.keras.layers.Input(shape=input_shape)
    
    if threshold is None:
        threshold = auto_threshold_factor / (input_shape[0] * input_shape[1])
        print('Auto threshold', threshold)
    
    p = i
            
    if concept_vector:
        f = tf.keras.layers.Flatten()(i)
        f = tf.keras.layers.Dense(p.shape[-1], activation='relu')(f)
        f = tf.reshape(f, [-1,1,1,f.shape[-1]])   
        
        p = p+f
        
    for c in extra_layers_channels:
        p = tf.keras.layers.Conv2D(c,1,activation='relu')(p)
    
    
    p = tf.keras.layers.Conv2D(1,1,activation='linear')(p)
    p = tf.keras.layers.Activation(activation)(p)
    p = p * pre_tau
    
    if sampling:
        if sigmoid:
            print('Warning: sigmoid option does not do anything while sampling')
        gs = []

        for _ in range(k):
          g = gumbel(tf.shape(p))
          g = global_softmax((g + p) * tau)
          gs.append(g)

        g = tf.stack(gs, axis=-1)
        g = tf.math.reduce_max(g, axis=-1)
        g = tf.expand_dims(g,-1)
        
    elif sigmoid:
        g = tf.math.sigmoid(p * tau)
    else:
        g = global_softmax(p * tau)
        g = tf.expand_dims(g,-1)
        
    if threshold > 0:
        g = g - threshold
        g = tf.keras.layers.Activation('relu')(g)

       
    if fixed_k_sampling:
        p_flat = tf.reshape(p, [-1, p.shape[1] * p.shape[2] * p.shape[3]])
        v, _ = tf.math.top_k(p_flat, k=k, sorted=True)
        v = v[:,-1]
        discrete_g = tf.cast(tf.greater_equal(tf.transpose(p),tf.transpose(v)),tf.float32)
        sum_vals = tf.math.reduce_sum(discrete_g,axis=(1,2))
        discrete_g = discrete_g / sum_vals
        discrete_g = tf.transpose(discrete_g)
    elif sigmoid:
        discrete_g = tf.cast(tf.greater_equal(tf.transpose(tf.math.sigmoid(p * tau)),.5),tf.float32)
        discrete_g = tf.transpose(discrete_g)
    else:
        max_vals = tf.math.reduce_max(p,axis=(1,2)) 
        discrete_g = tf.cast(tf.greater_equal(tf.transpose(p),tf.transpose(max_vals)),tf.float32)
        sum_vals = tf.math.reduce_sum(discrete_g,axis=(1,2))
        discrete_g = discrete_g / sum_vals
        discrete_g = tf.transpose(discrete_g)
        
    
    return tf.keras.Model(i,outputs=[g,p, discrete_g], name=name)


import os

class ModelWrapper():    
    
    def __init__(self, data_loader, conv_stack, x_block, name,
                 weights_dir='weights_dir/', input_shape=(224,224,3), prefix='', dropout=True):
        self.data_loader = data_loader
        self.conv_stack = conv_stack
        self.input_shape = input_shape
        self.num_classes = self.data_loader.num_classes
        self.x_block = x_block
        self.name = name
        self.weights_dir = weights_dir
        self.prefix = prefix
        self.dropout = dropout
        
        self.conv_stack.trainable = False
        
        self.base_head = self.make_head('mean', 'base_head')
        self.x_head = self.make_head('sum', 'x_head', dropout=self.dropout)
        
        
        i = tf.keras.layers.Input(shape=input_shape)
        features = self.conv_stack(i)
        x = self.base_head(features)
        self.base_model = tf.keras.Model(i,x)
                
        g, p, discrete_g = self.x_block(features)
        masked_features = features * g
        
        o = self.x_head(masked_features)
        self.x_model = tf.keras.Model(i,o)
        self.x_model_full = tf.keras.Model(i,outputs=[o,g,p])


        masked_features = features * discrete_g
        o = self.x_head(masked_features)
        self.x_model_discrete = tf.keras.Model(i,o)
        self.x_model_full_discrete = tf.keras.Model(i,outputs=[o,discrete_g,p])

        self.x_cam_model = tf.keras.Model(i,masked_features)
        
    def predict_base(self, imgs):
        return self.base_model.predict(imgs)
    
    def predict_features(self, imgs):
        return self.conv_stack.predict(imgs)
    
    def predict_x_soft(self, imgs):
        return self.x_model_full.predict(imgs)
    
    def predict_x_hard(self, imgs):
        return self.x_model_full_discrete.predict(imgs)
    
    def predict_x_features(self, imgs):
        return self.x_cam_model.predict(imgs)
        
    def compile_model(self, model):
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                           metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])
        
    def make_head(self, pooling_mode, name, dropout=True):        
        input_shape = self.conv_stack.output.shape[1:]
        
        i = tf.keras.layers.Input(shape=input_shape)
        if pooling_mode == 'mean':
            x = tf.math.reduce_mean(i, axis=(1,2))
        elif pooling_mode == 'sum':
            x = tf.math.reduce_sum(i, axis=(1,2))
        elif pooling_mode =='max':
            x = tf.math.reduce_max(i, axis=(1,2))
        else:
            assert(False)
        
        if dropout:
            x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.Dense(self.num_classes, name='logits')(x)
        x = tf.keras.layers.Activation('softmax')(x)  
        
        return tf.keras.Model(i,x, name=name)
    
    def head_file_name(self):
        return os.path.join(self.weights_dir, self.name + '_head.h5')
    
    def unfrozen_head_file_name(self):
        return os.path.join(self.weights_dir, self.name + '_' + self.prefix + 'x_unfrozen_head.h5')
    
    def x_file_name(self):
        return os.path.join(self.weights_dir, self.name + '_' + self.prefix + 'x_block.h5')
    
    def train_head(self,epochs, force_retrain=False):        
        self.compile_model(self.base_model)
        
        file = self.head_file_name()
        
        if (not os.path.exists(file)) or force_retrain:
            self.base_model.fit(self.data_loader.dataset['train'],
                                validation_data=None, epochs=epochs)
            self.base_head.save_weights(file)
        else:
            print('Reloading head weights from', file)
            self.base_head.load_weights(file)
            
    def train_x(self, epochs, force_retrain=False, train_head=False):                             
        self.compile_model(self.x_model)
            
        unfrozen_head_file = self.unfrozen_head_file_name()
        head_file = self.head_file_name()   
        x_file = self.x_file_name()        
        
        if (not os.path.exists(x_file)) or force_retrain:
            self.x_head.load_weights(head_file)
            self.x_head.trainable = train_head
            
            self.x_model.fit(self.data_loader.dataset['train'],
                                validation_data=None, epochs=epochs)
            
            self.x_block.save_weights(x_file)
            if train_head:
                self.x_head.save_weights(unfrozen_head_file)
                        
        else:
            print('Reloading x-block weights from', x_file)
            self.x_block.load_weights(x_file)
            if train_head:
                if (not os.path.exists(unfrozen_head_file)):
                    print('Cannot find unfrozen head file', unfrozen_head_file)
                    assert False
                print('Reloading unfrozen head weights from', unfrozen_head_file)
                self.x_head.load_weights(unfrozen_head_file)
            else:
                self.x_head.load_weights(head_file)
    
    def evaluate_base(self): 
        print('\nEvaluating uninterpretable baseline:')
        self.compile_model(self.base_model)
        return self.base_model.evaluate(self.data_loader.dataset['test'])
        
    def evaluate_x_soft(self): 
        print('\nEvaluating with soft explanations:')
        self.compile_model(self.x_model)
        return self.x_model.evaluate(self.data_loader.dataset['test'])
    
    def evaluate_x_hard(self):
        print('\nEvaluating with hard explanations:')                         
        self.compile_model(self.x_model_discrete)
        return self.x_model_discrete.evaluate(self.data_loader.dataset['test'])
        
    def evaluate_all(self):
        b = self.evaluate_base()
        s = self.evaluate_x_soft()
        h = self.evaluate_x_hard()
        return b, s, h
            
        
def make_fixed_size_block(conv_stack, use_threshold=False):
    if use_threshold:
        threshold = None
    else:
        threshold = 0

    return make_x_block(conv_stack.output.shape[1:],threshold=threshold,
        k=8,
        sampling=True,
        activation='linear',
        fixed_k_sampling=True)

def make_bounded_logit_block(conv_stack, use_threshold=False):
    if use_threshold:
        threshold = None
    else:
        threshold = 0
        
    return make_x_block(conv_stack.output.shape[1:],threshold=threshold)


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

conv_stack =  EfficientNetB0(include_top=False, input_shape=(224,224,3))

log_dir = 'logs'

def make_prefix(fixed_size, threshold, train_head, dropout=True):
    return ('auto_threshold_' if threshold else 'no_threshold_') +\
        ('fixed_size' if fixed_size else 'bounded_logit') +\
        ('trained_head' if train_head else 'frozen_head') +\
        ('' if dropout else 'no_dropout')

def make_model(ds, fixed_size, threshold, train_head, run=0, dropout=True):
    model_string = ds + str(run)
    prefix = make_prefix(fixed_size, threshold, train_head, dropout=dropout)
    if fixed_size:
        block = make_fixed_size_block(conv_stack,use_threshold=threshold)
    else:
        block = make_bounded_logit_block(conv_stack,use_threshold=threshold)

    loader = TfDataSetLoader(ds)
    wrapper = ModelWrapper(loader, conv_stack, block, model_string, prefix=prefix, dropout=dropout)
    return wrapper, model_string, prefix

def make_wrapper(ds,
              fixed_size,
              threshold,
              train_head,
              force_retrain=False):
    
    
    wrapper, model_string, prefix = make_model(ds, fixed_size, threshold, train_head)

    print('\n-----------------')
    print(model_string, prefix)
    print('-----------------\n')

    epochs = 2 if ds == 'cats_vs_dogs' else 5
    wrapper.train_head(epochs, force_retrain=force_retrain)
    wrapper.train_x(epochs, force_retrain=force_retrain, train_head=train_head)

    b, s, h = wrapper.evaluate_all()

    with open(os.path.join(log_dir, ds + '_' + prefix), "a") as myfile:
        myfile.write(','.join([','.join([str(x) for x in xs]) for xs in [b,s,h]]))
        myfile.write('\n')
        
    return wrapper


def load_model(ds, fixed_size, threshold, train_head, run=0):
    wrapper, _, _ = make_model(ds, fixed_size, threshold, train_head, run=run)
    
    wrapper.train_head(0, force_retrain=False)
    wrapper.train_x(0, force_retrain=False, train_head=train_head)
    
    return wrapper