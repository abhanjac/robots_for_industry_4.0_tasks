# -*- coding: utf-8 -*-
'''
This script describes a class for the cnn.

@author: abhanjac
'''

import tensorflow as tfl

from utils import *

class networkClassifier(object):
    '''
    Class that defines the tiny yolo model, along with its associated functions.
    '''
    def __init__(self):
        '''
        Initializes some of the fixed parameters of the model.
        '''
        # Defining methods to initilize the model weights and biases.
        self.initW = tfl.glorot_normal_initializer(dtype=tfl.float32)
        self.initB = tfl.zeros_initializer()
        
        # Dictionary to hold the individual outputs of each model layer.
        self.layerOut = {}
        # Flag that indicates if the output of the layers has to be saved in 
        # the dictionary or not.
        self.saveLayer = False
        
        # Defining the optimizer.
        # This is done here because we will be needing the optimizerName in the 
        # test function as well. If we will define the optimizer in the train
        # function, then during testing when the train function is not called,
        # the optimizerName will not be initialized. So it is defined in init
        # such that it gets defined as the class object is initialized.
        self.optimizer = tfl.train.AdamOptimizer(learning_rate=learningRate)
        # Name of optimizer ('Adam' in this case).
        self.optimizerName = self.optimizer.get_name()
        
        # This flag indicates if the network is in training mode or not.
        self.isTraining = False
        
#===============================================================================
        
    def model(self, x):
        '''
        This defines the overall network structure of the tiny yolo network.

        layer     filters   kernel      input                   output
        0 conv    8         3 x 3 / 1   480 x 640 x 3      ->   480 x 640 x 8
        1 max               2 x 2 / 2   480 x 640 x 8      ->   240 x 320 x 8
        2 conv    16        3 x 3 / 1   240 x 320 x 8      ->   240 x 320 x 16
        3 max               2 x 2 / 2   240 x 320 x 16     ->   120 x 160 x 16
        2a conv    32       3 x 3 / 1   120 x 160 x 16     ->   120 x 160 x 32
        3a max              2 x 2 / 2   120 x 160 x 32     ->   60 x 80 x 32
        4 conv    16        1 x 1 / 1   60 x 80 x 32       ->   60 x 80 x 16
        5 conv    128       3 x 3 / 1   60 x 80 x 16       ->   60 x 80 x 128
        8 max               2 x 2 / 2   60 x 80 x 128      ->   30 x 40 x 128
        9 conv    32        1 x 1 / 1   30 x 40 x 128      ->   30 x 40 x 32
        10 conv   256       3 x 3 / 1   30 x 40 x 32       ->   30 x 40 x 256
        13 max              2 x 2 / 2   30 x 40 x 256      ->   15 x 20 x 256
        14 conv   64        1 x 1 / 1   15 x 20 x 256      ->   15 x 20 x 64
        15 conv   512       3 x 3 / 1   15 x 20 x 64       ->   15 x 20 x 512
        18 conv   128       1 x 1 / 1   15 x 20 x 512      ->   15 x 20 x 128
        19 conv   nClasses  1 x 1 / 1   15 x 20 x 128      ->   15 x 20 x nClasses
        20 avg                          15 x 20 x nClasses ->   nClasses

        The final softmax and cost are implemented in the loss function.
        '''

        x = tfl.convert_to_tensor(x, dtype=tfl.float32)
        
        # Input size 480 x 640 x 3 (H x W x D).
        layerIdx = '0'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=8, strides=1, padding='SAME', \
                             name=layerName, bias_initializer=self.initB, \
                             kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 480 x 640 x 8 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 480 x 640 x 8 (H x W x D).
        layerIdx = '1'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 240 x 320 x 8 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 240 x 320 x 8 (H x W x D).
        layerIdx = '2'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=16, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 240 x 320 x 16 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 240 x 320 x 16 (H x W x D).
        layerIdx = '3'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 120 x 160 x 16 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 120 x 160 x 16 (H x W x D).
        layerIdx = '2a'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 120 x 160 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 120 x 160 x 32 (H x W x D).
        layerIdx = '3a'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 60 x 80 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 60 x 80 x 32 (H x W x D).
        layerIdx = '4'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=16, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 60 x 80 x 16 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 60 x 80 x 16 (H x W x D).
        layerIdx = '5'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 60 x 80 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 60 x 80 x 128 (H x W x D).
        layerIdx = '8'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 30 x 40 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 30 x 40 x 128 (H x W x D).
        layerIdx = '9'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 30 x 40 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 30 x 40 x 32 (H x W x D).
        layerIdx = '10'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=256, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 30 x 40 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 30 x 40 x 256 (H x W x D).
        layerIdx = '13'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 15 x 20 x 256 (H x W x D).
        layerIdx = '14'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=64, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 64 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 15 x 20 x 64 (H x W x D).
        layerIdx = '15'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=512, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 512 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 15 x 20 x 512 (H x W x D).
        layerIdx = '18'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 15 x 20 x 128 (H x W x D).
        layerIdx = '19'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=nClasses, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x nClasses (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 15 x 20 x nClasses (H x W x D).
        layerIdx = '20'
        layerName = 'pooling' + layerIdx
        # This next layer will behave as the global average pooling layer. So 
        # the padding has to be 'VALID' here to give a 1x1 output size. 
        x = tfl.layers.average_pooling2d(x, pool_size=(finalLayerH, finalLayerW), strides=1, \
                                        padding='VALID', name=layerName)
        
        x = tfl.layers.flatten(x)    # This will keep the 0th dimension 
        # (batch size dimension) intact and flatten the rest of the elements 
        # (which has shape 1 x 1 x nClasses right now) into a single dimension 
        # (of size nClasses).

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size nClasses.
        
        return x
    
#===============================================================================

    def loss(self, logits, labels):
        '''
        Defines the prediction loss. This is the loss only for the first phase 
        of the training, when the network is only trained for classification
        and not for prediction of bbox.
        The input labels should be in multi hot vector form.
        '''
        labels = tfl.convert_to_tensor(labels)
        labels = tfl.cast(labels, dtype=tfl.float32)

        # The sigmoid_cross_entropy_with_logits loss is chosen as it measures 
        # probability error in classification tasks where each class is 
        # independent and not mutually exclusive. E.g. one could perform 
        # multilabel classification where a picture can contain objects of two 
        # classes.
        lossTensor = tfl.losses.sigmoid_cross_entropy(logits=logits, \
                                        multi_class_labels=labels, weights=1.0)
        
        # Return the average loss over this batch.
        #return tfl.reduce_sum(lossTensor)
        return tfl.reduce_mean(lossTensor)
        
#==============================================================================

    def train(self, trainDir=None, validDir=None):
        '''
        Trains the model.
        '''
        if trainDir is None or validDir is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in train. Aborting.\n')
            sys.exit() 
        
        # SET INPUTS AND LABELS.
        
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches (especially the last batch).
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        # Labels are multi hot vectors.
        y = tfl.placeholder(dtype=tfl.int32, name='yPlaceholder', \
                            shape=[None, nClasses])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.
            predProb = tfl.sigmoid(predLogits)    # Predicted probabilities.
            
            # List of model variables.
            listOfModelVars = []
            for v in tfl.global_variables():     listOfModelVars.append(v)

            #for v in listOfModelVars:
                #print('Model: {}, Variable: {}'.format(modelName, v))
            
#-------------------------------------------------------------------------------
        
        # CALCULATE LOSS.
        
        loss = self.loss(logits=predLogits, labels=y)
        
        # DEFINE OPTIMIZER AND PERFORM BACKPROP.
        optimizer = self.optimizer
        
        # While executing an operation (such as trainStep), only the subgraph 
        # components relevant to trainStep will be executed. The 
        # update_moving_averages operation (for the batch normalization layers) 
        # is not a parent of trainStep in the computational graph, so it will 
        # never update the moving averages by default. To get around this, 
        # we have to explicitly tell the graph in the following manner.
        update_ops = tfl.get_collection(tfl.GraphKeys.UPDATE_OPS)
        with tfl.control_dependencies(update_ops):
            trainStep = optimizer.minimize(loss)

        # List of optimizer variables.
        listOfOptimizerVars = list(set(tfl.global_variables()) - set(listOfModelVars))

        #for v in listOfOptimizerVars:
            #print('Optimizer: {}, Variable: {}'.format(self.optimizerName, v))

#-------------------------------------------------------------------------------
        
        # CREATE A LISTS TO HOLD ACCURACY AND LOSS VALUES.
        
        # This list will have strings for each epoch. Each of these strings 
        # will be like the following:
        # 'epoch, learningRate, trainLoss, trainAcc (%), validLoss, validAcc (%)'
        statistics = []
        
        # Format of the statistics values.
        statisticsFormat = 'epoch, learningRate, batchSize, trainLoss, trainAcc (%), ' \
                           'validLoss, validAcc (%), epochProcessTime'
                        
#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        print('\nStarting session. Optimizer: {}, Learning Rate: {}, ' \
               'Batch Size: {}'.format(self.optimizerName, learningRate, batchSize))
        
        self.isTraining = True    # Enabling the training flag. 
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPath, \
                                                              training=self.isTraining)
        startEpoch = latestEpoch + 1    # Start from the next epoch.

#-------------------------------------------------------------------------------
        
        # LOAD PARAMETERS FROM CHECKPOINTS.
        
        if ckptPath != None:    # Only when some checkpoint is found.
            with open(jsonFilePath, 'r') as infoFile:
                infoDict = json.load(infoFile)
                
            if infoDict['learningRate'] == learningRate and infoDict['batchSize'] == batchSize:
                # Since all old variables will be loaded here, so we do not need
                # to initialize any other variables.
                
                # Now define a new saver with all the variables.
                saver = tfl.train.Saver(listOfModelVars + listOfOptimizerVars)
                saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.
                saver.restore(sess, ckptPath)

                print('\nReloaded ALL variables from checkpoint: {}\n'.format(\
                                                            ckptPath))

#-------------------------------------------------------------------------------

            else:
                # else load only weight and biases and skip optimizer 
                # variables. Otherwise there will be errors.
                # So define the saver with the list of only model variables.
                saver = tfl.train.Saver(listOfModelVars)
                saver.restore(sess, ckptPath)
                # Since only the model variables will be loaded here, so we
                # have to initialize other variables (optimizer variables)
                # separately.
                sess.run(tfl.variables_initializer(listOfOptimizerVars))

                # The previous saver will only save the listOfModelVars
                # as it is defined using only those variables (as the 
                # checkpoint can only give us those values as valid for
                # restoration). But now since we have all the varibles 
                # loaded and all the new ones initialized, so we redefine 
                # the saver to include all the variables. So that while 
                # saving in the end of the epoch, it can save all the 
                # variables (and not just the listOfModelVars).
                saver = tfl.train.Saver(listOfModelVars + listOfOptimizerVars)
                saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.

                # Load mean and std values.
                mean, std = infoDict['mean'], infoDict['std']
                
                print('\nCurrent parameters:\nlearningRate: {}, batchSize: {}' \
                       '\nPrevious parameters (inside checkpoint {}):\nlearningRate: ' \
                       '{}, batchSize: {}\nThey are different.\nSo reloaded only ' \
                       'MODEL variables from checkpoint: {}\nAnd initialized ' \
                       'other variables.\n'.format(learningRate, batchSize, ckptPath, \
                       infoDict['learningRate'], infoDict['batchSize'], ckptPath))
                
            # Reloading accuracy and loss statistics, mean and std from checkpoint.
            statistics = infoDict['statistics']
            mean = np.array(infoDict['mean'])
            std = np.array(infoDict['std'])
            maxValidAcc = infoDict['maxValidAcc']
            minValidLoss = infoDict['minValidLoss']
            
            # If the batchsize changes, then the minValidLoss should also be 
            # scaled according to that.
            if batchSize != infoDict['batchSize']:
                minValidLoss = infoDict['minValidLoss'] * batchSize / infoDict['batchSize']

#-------------------------------------------------------------------------------

        else:
            # When there are no valid checkpoints initialize the saver to 
            # save all parameters.
            saver = tfl.train.Saver(listOfModelVars + listOfOptimizerVars)
            saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.

            sess.run(tfl.variables_initializer(listOfModelVars + listOfOptimizerVars))
           
            # Calculate mean and std.
            if recordedMean is None or recordedStd is None:
                mean, std = datasetMeanStd(trainDir)
                print('\nMean of {}: {}'.format(trainDir, mean))
                print('\nStd of {}: {}'.format(trainDir, std))
            else:
                mean, std = recordedMean, recordedStd

            maxValidAcc = 0.0
            minValidLoss = np.inf

#-------------------------------------------------------------------------------
        
        # TRAINING AND VALIDATION.
        
        print('\nStarted Training...\n')
        
        for epoch in range(startEpoch, nEpochs+1):
            # epoch will be numbered from 1 to 150 if there are 150 epochs.
            # Is is not numbered from 0 to 149.

            epochProcessTime = time.time()
            
            # TRAINING PHASE.
            
            self.isTraining = True    # Enabling training flag at the start of 
            # the training phase of each epoch as it will be disabled in the 
            # corresponding validaton phase. 

            # This list contains the filepaths for all the images in trainDir.
            # If there are unwanted files like '.thumbs' etc. then those are 
            # filtered as well.
            listOfRemainingTrainImg = [i for i in os.listdir(\
                                        os.path.join(trainDir, 'images')) \
                                                if i[-3:] == 'png']

            nTrainImgs = len(listOfRemainingTrainImg)
            nTrainBatches = int(np.ceil(nTrainImgs / batchSize))
            
            # Interval at which the status will be printed in the terminal.
            printInterval = 1 if nTrainBatches < 10 else int(nTrainBatches / 10)
            
            trainLoss, trainAcc = 0.0, 0.0
            trainBatchIdx = 0    # Counts the number of batch processed.
            
            # Storing information of current epoch for recording later in statistics.
            # This is recorded as a string for better visibility in the json file.
            currentStatistics = '{}, {}, {}, '.format(epoch, learningRate, batchSize)
            
#-------------------------------------------------------------------------------
            
            # Scan entire training dataset.
            while len(listOfRemainingTrainImg) > 0:

                trainBatchProcessTime = time.time()

                # Creating batches. Also listOfRemainingTrainImg is updated.
                trainImgBatch, trainLabelBatch, listOfRemainingTrainImg, _ = \
                    createBatchForClassification(trainDir, listOfRemainingTrainImg, \
                                                  batchSize, shuffle=True)
                
                feedDict = {x: trainImgBatch, y: trainLabelBatch}
                if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                    trainLayerOut = sess.run(self.layerOut, feed_dict=feedDict)

                trainPredLogits = sess.run(predLogits, feed_dict=feedDict)
                trainBatchLoss = sess.run(loss, feed_dict=feedDict)
                sess.run(trainStep, feed_dict=feedDict)
                
                trainLoss += (trainBatchLoss / nTrainBatches)
                
                # The trainPredLogits is an array of logits. It needs to be
                # converted to sigmoid to get probability and then we need
                # to extract the max index to get the labels.
                trainPredProb = sess.run(predProb, feed_dict=feedDict)

                # If the probability is more than the threshold, the 
                # corresponding label element is considered as 1 else 0.
                trainPredLabel = np.asarray(trainPredProb > threshProb, dtype=np.int32)
                
#-------------------------------------------------------------------------------                
                
                matches = np.array(trainPredLabel == trainLabelBatch, dtype=np.int32)
                
                # This will be an array of batchSize x 6 (if there are 6 classes).
                # Since independent probabilities are calculated, so there may be
                # more than one 1's or 0's among these 6 elements.
                # Hence, a completely correct prediction will have a True match 
                # for all of these 6 elements. The sum of this matches array is
                # calculated along the columns. If that results in 6 then it 
                # means that a perfect match has happened.
                matches = np.sum(matches, axis=1)
                perfectMatch = np.asarray(np.ones(matches.shape) * nClasses, dtype=np.int32)
                matches1 = np.array(matches == perfectMatch, dtype=np.int32)
                
                trainAcc += (100*np.sum(matches1)) / nTrainImgs
                
                trainBatchIdx += 1
                trainBatchProcessTime = time.time() - trainBatchProcessTime

                # Print the status on the terminal every 10 batches.
                if trainBatchIdx % printInterval == 0:
                    print('Epoch: {}/{},\tBatch: {}/{},\tBatch loss: ' \
                           '{:0.6f},\tProcess time for {} batch: {}'.format(epoch, \
                           nEpochs, trainBatchIdx, nTrainBatches, trainBatchLoss, \
                           printInterval, prettyTime(trainBatchProcessTime*printInterval)))
            
            # Recording training loss and accuracy in current statistics string.
            currentStatistics += '{}, {}, '.format(trainLoss, trainAcc)
            
#-------------------------------------------------------------------------------
            
            # VALIDATION PHASE.
            
            self.isTraining = False    # Disabling the training flag.
            
            # This list contains the filepaths for all the images in validDir.
            # If there are unwanted files like '.thumbs' etc. then those are 
            # filtered as well.
            listOfRemainingValidImg = [i for i in os.listdir(\
                                        os.path.join(validDir, 'images')) \
                                                if i[-3:] == 'png']
            
            nValidImgs = len(listOfRemainingValidImg)
            nValidBatches = int(np.ceil(nValidImgs / batchSize))
            
            # Interval at which the status will be printed in the terminal.
            printInterval = 1 if nValidBatches < 3 else int(nValidBatches / 3)

            validLoss, validAcc = 0.0, 0.0
            validBatchIdx = 0    # Counts the number of batch processed.
            
            print('\n\nValidation phase for epoch {}.\n'.format(epoch))
            
#-------------------------------------------------------------------------------
            
            # Scan entire validation dataset.
            while len(listOfRemainingValidImg) > 0:
                
                validBatchProcessTime = time.time()

                # Creating batches. Also listOfRemainingValidImg is updated.
                # The shuffle is off for validation and the mean and std are
                # the same as calculated on the training set.
                validImgBatch, validLabelBatch, listOfRemainingValidImg, _ = \
                    createBatchForClassification(validDir, listOfRemainingValidImg, \
                                                  batchSize, shuffle=False)
                
                feedDict = {x: validImgBatch, y: validLabelBatch}
                if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                    validLayerOut = sess.run(self.layerOut, feed_dict=feedDict)
                
                validPredLogits = sess.run(predLogits, feed_dict=feedDict)
                validBatchLoss = sess.run(loss, feed_dict=feedDict)
                
                validLoss += (validBatchLoss / nValidBatches)
                                    
                # The validPredLogits is an array of logits. It needs to be 
                # converted to sigmoid to get probability and then we need
                # to extract the max index to get the labels.
                validPredProb = sess.run(predProb, feed_dict=feedDict)
                
                # If the probability is more than the threshold, the 
                # corresponding label element is considered as 1 else 0.
                validPredLabel = np.asarray(validPredProb > threshProb, dtype=np.int32)
                
#------------------------------------------------------------------------------- 
                
                matches = np.array(validPredLabel == validLabelBatch, dtype=np.int32)

                # This will be an array of batchSize x 5 (as there are 5 classes).
                # Since independent probabilities are calculated, so there may be
                # more than one 1's or 0's among these 5 elements.
                # Hence, a completely correct prediction will have a True match 
                # for all of these 5 elements. The sum of this matches array is
                # calculated along the columns. If that results in 5 then it 
                # means that a perfect match has happened.
                matches = np.sum(matches, axis=1)
                perfectMatch = np.asarray(np.ones(matches.shape) * nClasses, dtype=np.int32)
                matches1 = np.array(matches == perfectMatch, dtype=np.int32)

                validAcc += (100*np.sum(matches1)) / nValidImgs
                
                validBatchIdx += 1
                validBatchProcessTime = time.time() - validBatchProcessTime

                # Print the status on the terminal every 10 batches.
                if validBatchIdx % printInterval == 0:     
                    print('Epoch: {}/{},\tBatch: {}/{},\tBatch loss: ' \
                           '{:0.6f},\tProcess time for {} batch: {}'.format(epoch, \
                           nEpochs, validBatchIdx, nValidBatches, validBatchLoss, \
                           printInterval, prettyTime(validBatchProcessTime*printInterval)))

            # Recording validation accuracy in current statistics string.
            currentStatistics += '{}, {}, '.format(validLoss, validAcc)

#-------------------------------------------------------------------------------

            # STATUS UPDATE.
            
            epochProcessTime = time.time() - epochProcessTime
            
            # Recording epoch processing time in current statistics string.
            currentStatistics += '{}'.format(prettyTime(epochProcessTime))
            
            # Noting accuracy after the end of all batches.
            statistics.append(currentStatistics)
            
            # Printing current epoch.
            print('\nEpoch {} done. Epoch time: {}, Train ' \
                   'loss: {:0.6f}, Train Accuracy: {:0.3f} %, Valid loss: {:0.6f}, ' \
                   'Valid Accuracy: {:0.3f} %\n'.format(epoch, \
                   prettyTime(epochProcessTime), trainLoss, trainAcc, validLoss, \
                   validAcc))

            # Saving the variables at some intervals, only if there is 
            # improvement in validation accuracy.
            if epoch % modelSaveInterval == 0 and validAcc > maxValidAcc:
                ckptSavePath = os.path.join(ckptDirPath, savedCkptName)
                saver.save(sess, save_path=ckptSavePath, global_step=epoch)
                
                maxValidAcc = validAcc      # Updating the maxValidAcc.
                minValidLoss = validLoss    # Updating the minValidLoss.
                
                # Saving the important info like learning rate, batch size,
                # and training error for the current epoch in a json file.
                # Converting the mean and std into lists before storing as
                # json cannot store numpy arrays. And also saving the training
                # and validation loss and accuracy statistics.
                jsonInfoFilePath = ckptSavePath + '-' + str(epoch) + '.json'
                with open(jsonInfoFilePath, 'w') as infoFile:
                    infoDict = {'epoch': epoch, 'batchSize': batchSize, \
                                 'learningRate': learningRate, \
                                 'mean': list(mean), 'std': list(std), \
                                 'maxValidAcc': maxValidAcc, \
                                 'minValidLoss': minValidLoss, \
                                 'statisticsFormat': statisticsFormat, \
                                 'statistics': statistics}
                    
                    json.dump(infoDict, infoFile, sort_keys=False, \
                               indent=4, separators=(',', ': '))
                
                print('Checkpoint saved.\n')

            # Updating the maxValidAcc value.
            elif validAcc > maxValidAcc:
                maxValidAcc = validAcc
                minValidLoss = validLoss
            
#-------------------------------------------------------------------------------
        
        self.isTraining = False   # Indicates the end of training.
        print('\nTraining completed with {} epochs.'.format(nEpochs))

        sess.close()        # Closing the session.
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.

#===============================================================================

    def test(self, testDir=None):
        '''
        Tests the model.
        '''
        if testDir is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in test. Aborting.\n')
            sys.exit() 

        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS AND LABELS.
        
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches (especially the last batch).
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        # Labels are multi hot vectors.
        y = tfl.placeholder(dtype=tfl.int32, name='yPlaceholder', \
                            shape=[None, nClasses])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.
            predProb = tfl.sigmoid(predLogits)    # Predicted probabilities.
            
            # List of model variables.
            listOfModelVars = []
            for v in tfl.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find(modelName) >= 0:
                    listOfModelVars.append(v)
                    #print('Model: {}, Variable: {}'.format(modelName, v))

#-------------------------------------------------------------------------------

        # CALCULATE LOSS.
        
        loss = self.loss(logits=predLogits, labels=y)

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPath, \
                                                              training=self.isTraining)

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tfl.train.Saver(listOfModelVars)
            saver.restore(sess, ckptPath)            
            print('\nReloaded ALL variables from checkpoint: {}'.format(\
                                                        ckptPath))
        else:
            # When there are no valid checkpoints.
            print('\nNo valid checkpoints found. Aborting.\n')
            return

        with open(jsonFilePath, 'r') as infoFile:
            infoDict = json.load(infoFile)

        # Reloading mean and std from checkpoint.
        mean = np.array(infoDict['mean'])
        std = np.array(infoDict['std'])

#-------------------------------------------------------------------------------
                    
        print('\nStarted Testing...\n')
            
        # TESTING PHASE.

        testingTime = time.time()

        # This list contains the filepaths for all the images in validDir.
        # If there are unwanted files like '.thumbs' etc. then those are 
        # filtered as well.
        listOfRemainingTestImg = [i for i in os.listdir(\
                                  os.path.join(testDir, 'images')) \
                                            if i[-3:] == 'png']
        
        nTestImgs = len(listOfRemainingTestImg)
        nTestBatches = int(np.ceil(len(listOfRemainingTestImg) / batchSize))
        
        # Interval at which the status will be printed in the terminal.
        printInterval = 1 if nTestBatches < 3 else int(nTestBatches / 3)
        
        testLoss, testAcc = 0.0, 0.0
        testBatchIdx = 0    # Counts the number of batches processed.

#-------------------------------------------------------------------------------

        # Just like the totalClsInstances, this totalDetectedClsInstances indicates
        # the number of times the class objects has been detected properly in the
        # images. This two will be later used to identify the individual class
        # detection accuracy.
        totalDetectedClsInstances = np.zeros(nClasses)

#-------------------------------------------------------------------------------
        
        # Scan entire validation dataset.
        while len(listOfRemainingTestImg) > 0:
        
            testBatchProcessTime = time.time()

            # Creating batches. Also listOfRemainingTestImg is updated.
            # The shuffle is off for validation and the mean and std are
            # the same as calculated on the training set.
            testImgBatch, testLabelBatch, listOfRemainingTestImg, listOfSelectedTestImg = \
                createBatchForClassification(testDir, listOfRemainingTestImg, \
                                              batchSize, shuffle=False)
            
            feedDict = {x: testImgBatch, y: testLabelBatch}
            if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                testLayerOut = sess.run(self.layerOut, feed_dict=feedDict)
            
            testPredLogits = sess.run(predLogits, feed_dict=feedDict)
            testBatchLoss = sess.run(loss, feed_dict=feedDict)
                
            testLoss += (testBatchLoss / nTestBatches)
                                                            
            # The testPredLogits is an array of logits. It needs to be 
            # converted to sigmoid to get probability and then we need
            # to extract the max index to get the labels.
            testPredProb = sess.run(predProb, feed_dict=feedDict)
            
            # If the probability is more than the threshold, the 
            # corresponding label element is considered as 1 else 0.
            testPredLabel = np.asarray(testPredProb > threshProb, dtype=np.int32)
                
#------------------------------------------------------------------------------- 
                
            matches = np.array(testPredLabel == testLabelBatch, dtype=np.int32)
            
            # Counting the number of class instances that were predicted properly 
            # in the current batch.
            nDetectedClsInstances = np.sum(matches, axis=0)
            totalDetectedClsInstances += nDetectedClsInstances
            
            # This will be an array of batchSize x 5 (as there are 5 classes).
            # Since independent probabilities are calculated, so there may be
            # more than one 1's or 0's among these 5 elements.
            # Hence, a completely correct prediction will have a True match 
            # for all of these 5 elements. The sum of this matches array is
            # calculated along the columns. If that results in 5 then it 
            # means that a perfect match has happened.
            matches = np.sum(matches, axis=1)
            perfectMatch = np.asarray(np.ones(matches.shape) * nClasses, dtype=np.int32)
            matches1 = np.array(matches == perfectMatch, dtype=np.int32)

            testAcc += (100*np.sum(matches1)) / (nTestBatches*batchSize)
            
            testBatchIdx += 1
            testBatchProcessTime = time.time() - testBatchProcessTime

            # Printing current status of testing.
            # print the status on the terminal every 10 batches.
            if testBatchIdx % printInterval == 0:
                print('Batch: {}/{},\tBatch loss: {:0.6f},\tProcess time for {} ' \
                       'batch: {}'.format(testBatchIdx, nTestBatches, testBatchLoss, \
                        printInterval, prettyTime(testBatchProcessTime*printInterval)))

#-------------------------------------------------------------------------------

            # Creating a list of images which have been misclassified.            
            if testBatchIdx == 1:
                misclassificationList = [m for mdx, m in enumerate(\
                                          listOfSelectedTestImg) if matches1[mdx] == 0]
            else:
                misclassificationList += [m for mdx, m in enumerate(\
                                           listOfSelectedTestImg) if matches1[mdx] == 0]

#-------------------------------------------------------------------------------
            
        # Calculate the individual class prediction accuracy.
        clsDetectionAcc = (totalDetectedClsInstances / nTestImgs) * 100

        # Printing the detection accuracy of individual classes.
        print('\n')
        for cdx in range(nClasses):
            print('Detection Accuracy for class {}: {:0.3f} %'.format(classIdxToName[cdx], \
                                                         clsDetectionAcc[cdx]))
        
#-------------------------------------------------------------------------------

        testingTime = time.time() - testingTime
        print('\n\nTesting done. Test Loss: {:0.6f}, Test Accuracy: {:0.3f} %, ' \
               'Testing time: {}'.format(testLoss, testAcc, prettyTime(testingTime)))
        
        # Save the misclassified examples list in a json file (use json.dumps 
        # not json.dump).
        with open('misclassified_example_list_{}.json'.format(testDir), 'w') as infoFile:
            infoDict = misclassificationList
            json.dump(misclassificationList, infoFile, indent=4, separators=(',', ':'))
        
        sess.close()        # Closing the session.
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.
        
        return testLoss, testAcc, testingTime

#===============================================================================

    def batchInference(self, imgBatch):
        '''
        This function evaluates the output of the model on an unknown batch of
        images (4d numpy array) and returns the predicted labels as a batch as well.
        '''
        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS.
        
        b, h, w, _ = imgBatch.shape

        # All the images in the batch are already resized to the appropriate shape 
        # when the batch was created, hence no need to resize again.
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.
            predProb = tfl.sigmoid(predLogits)    # Predicted probabilities.
            
            # List of model variables.
            listOfModelVars = []
            for v in tfl.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find(modelName) >= 0:
                    listOfModelVars.append(v)
                    #print('Model: {}, Variable: {}'.format(modelName, v))

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPath, \
                                                              training=self.isTraining)

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tfl.train.Saver(listOfModelVars)
            saver.restore(sess, ckptPath)            
#            print('\nReloaded ALL variables from checkpoint: {}'.format(\
#                                                        ckptPath))
        else:
            # When there are no valid checkpoints.
            print('\nNo valid checkpoints found. Aborting.\n')
            return

        with open(jsonFilePath, 'r') as infoFile:
            infoDict = json.load(infoFile)
                
        # Reloading mean and std from checkpoint.
        mean = np.array(infoDict['mean'])
        std = np.array(infoDict['std'])

#-------------------------------------------------------------------------------

#        # Normalizing by mean and std as done in case of training.
#        imgBatch = (imgBatch - mean) / std

        # Converting image to range 0 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        imgBatch = np.asarray(imgBatch, dtype=np.float32) / 127.5 - 1.0
        
        feedDict = {x: imgBatch}
        if self.saveLayer:    # Evaluate layer outputs if this flag is True.
            inferLayerOut = sess.run(self.layerOut, feed_dict=feedDict)
        else:   inferLayerOut = None

        inferPredLogits = sess.run(predLogits, feed_dict=feedDict)
                                                                                        
        # The testPredLogits is an array of logits. It needs to be 
        # converted to sigmoid to get probability and then we need
        # to extract the max index to get the labels.
        inferPredProb = sess.run(predProb, feed_dict=feedDict)
        
        # If the probability is more than the threshold, the 
        # corresponding label element is considered as 1 else 0.
        inferPredLabel = np.asarray(inferPredProb > threshProb, dtype=np.int32)

#-------------------------------------------------------------------------------
        
        sess.close()
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.

        return inferLayerOut, inferPredLogits, inferPredProb, inferPredLabel, mean, std
    
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================


class networkDetector(object):
    '''
    Class that defines the tiny yolo model, along with its associated functions.
    This model is for DETECTION. It can only be trained once the classification
    model has been trained already.
    '''
    def __init__(self):
        '''
        Initializes some of the fixed parameters of the model.
        '''
        # Defining methods to initilize the model weights and biases.
        self.initW = tfl.glorot_normal_initializer(dtype=tfl.float32)
        self.initB = tfl.zeros_initializer()
        
        # Dictionary to hold the individual outputs of each model layer.
        self.layerOut = {}
        # Flag that indicates if the output of the layers has to be saved in 
        # the dictionary or not.
        self.saveLayer = False
        
        # Defining the optimizer.
        # This is done here because we will be needing the optimizerName in the 
        # test function as well. If we will define the optimizer in the train
        # function, then during testing when the train function is not called,
        # the optimizerName will not be initialized. So it is defined in init
        # such that it gets defined as the class object is initialized.
        self.optimizer = tfl.train.AdamOptimizer(learning_rate=learningRate)
        # Name of optimizer ('Adam' in this case).
        self.optimizerName = self.optimizer.get_name()
        
        # This flag indicates if the network is in training mode or not.
        self.isTraining = False
        
        # This id will be appended to the names of the layers which are newly
        # added for the detection phase. This is needed as at some point of time
        # we need to selectively initialize only these layers, excluding the 
        # other layers which are common with the classifier.
        self.detectorLayerID = 'Detector'
        
#===============================================================================
        
    def model(self, x):
        '''
        This defines the overall network structure of the tiny yolo network.

        layer     filters   kernel      input                   output
        0 conv    8         3 x 3 / 1   480 x 640 x 3      ->   480 x 640 x 8
        1 max               2 x 2 / 2   480 x 640 x 8      ->   240 x 320 x 8
        2 conv    16        3 x 3 / 1   240 x 320 x 8      ->   240 x 320 x 16
        3 max               2 x 2 / 2   240 x 320 x 16     ->   120 x 160 x 16
        2a conv    32       3 x 3 / 1   120 x 160 x 16     ->   120 x 160 x 32
        3a max              2 x 2 / 2   120 x 160 x 32     ->   60 x 80 x 32
        4 conv    16        1 x 1 / 1   60 x 80 x 32       ->   60 x 80 x 16
        5 conv    128       3 x 3 / 1   60 x 80 x 16       ->   60 x 80 x 128
        8 max               2 x 2 / 2   60 x 80 x 128      ->   30 x 40 x 128
        9 conv    32        1 x 1 / 1   30 x 40 x 128      ->   30 x 40 x 32
        10 conv   256       3 x 3 / 1   30 x 40 x 32       ->   30 x 40 x 256
        13 max              2 x 2 / 2   30 x 40 x 256      ->   15 x 20 x 256
        14 conv   64        1 x 1 / 1   15 x 20 x 256      ->   15 x 20 x 64
        15 conv   512       3 x 3 / 1   15 x 20 x 64       ->   15 x 20 x 512
        18 conv   128       1 x 1 / 1   15 x 20 x 512      ->   15 x 20 x 128
        21 conv   128       3 x 3 / 1   15 x 20 x 128      ->   15 x 20 x 128
        22 conv   128       3 x 3 / 1   15 x 20 x 128      ->   15 x 20 x 128
        23 conv   128       3 x 3 / 1   15 x 20 x 128      ->   15 x 20 x 128
        24 conv   nAnchor * (nClasses + 5)  
                        1 x 1 / 1   15 x 20 x 128   ->  15 x 20 x nAnchor * (nClasses + 5)
           reshape      15 x 20 x nAnchor * (nClasses + 5)  ->  15 x 20 x nAnchor x (nClasses + 5)
        
        The input to the 6 channel classifier layer has been connected to a 
        series of 3 conv layers that has 128 filters each and are all 3x3 conv 
        layers. Then after that there is a final layer of 1x1 conv layer with 
        15 x 20 x nAnchor x (nClasses + 5) no. of filters. The cost function is 
        implemented inside the loss function.
        '''
        
        x = tfl.convert_to_tensor(x, dtype=tfl.float32)
        
        # Input size 480 x 640 x 3 (H x W x D).
        layerIdx = '0'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=8, strides=1, padding='SAME', \
                             name=layerName, bias_initializer=self.initB, \
                             kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 480 x 640 x 8 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 480 x 640 x 8 (H x W x D).
        layerIdx = '1'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 240 x 320 x 8 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 240 x 320 x 8 (H x W x D).
        layerIdx = '2'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=16, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 240 x 320 x 16 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 240 x 320 x 16 (H x W x D).
        layerIdx = '3'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 120 x 160 x 16 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 120 x 160 x 16 (H x W x D).
        layerIdx = '2a'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 120 x 160 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 120 x 160 x 32 (H x W x D).
        layerIdx = '3a'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 60 x 80 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 60 x 80 x 32 (H x W x D).
        layerIdx = '4'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=16, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 60 x 80 x 16 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 60 x 80 x 16 (H x W x D).
        layerIdx = '5'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 60 x 80 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 60 x 80 x 128 (H x W x D).
        layerIdx = '8'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 30 x 40 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 30 x 40 x 128 (H x W x D).
        layerIdx = '9'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 30 x 40 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 30 x 40 x 32 (H x W x D).
        layerIdx = '10'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=256, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 30 x 40 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 30 x 40 x 256 (H x W x D).
        layerIdx = '13'
        layerName = 'pooling' + layerIdx
        x = tfl.layers.max_pooling2d(x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 15 x 20 x 256 (H x W x D).
        layerIdx = '14'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=64, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 64 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 15 x 20 x 64 (H x W x D).
        layerIdx = '15'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=512, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)

        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 512 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 15 x 20 x 512 (H x W x D).
        layerIdx = '18'
        layerName = 'conv' + layerIdx
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)
        
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 15 x 20 x 128 (H x W x D).
        layerIdx = '21'
        layerName = 'conv' + layerIdx + self.detectorLayerID
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx + self.detectorLayerID
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx + self.detectorLayerID
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)
        
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 15 x 20 x 128 (H x W x D).
        layerIdx = '22'
        layerName = 'conv' + layerIdx + self.detectorLayerID
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx + self.detectorLayerID
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx + self.detectorLayerID
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)
        
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 15 x 20 x 128 (H x W x D).
        layerIdx = '23'
        layerName = 'conv' + layerIdx + self.detectorLayerID
        x = tfl.layers.conv2d(x, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx + self.detectorLayerID
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx + self.detectorLayerID
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)
        
        if self.saveLayer:  self.layerOut[layerName] = x
        
        # Output size 15 x 20 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 15 x 20 x 128 (H x W x D).
        layerIdx = '24'
        layerName = 'conv' + layerIdx + self.detectorLayerID
        x = tfl.layers.conv2d(x, kernel_size=(1,1), filters=nAnchors*(nClasses+5), \
                              padding='SAME', name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'activation' + layerIdx + self.detectorLayerID
        x = tfl.nn.leaky_relu(x, alpha=leak, name=layerName)
        if self.saveLayer:  self.layerOut[layerName] = x
        
        layerName = 'batchNorm' + layerIdx + self.detectorLayerID
        x = tfl.layers.batch_normalization(x, training=self.isTraining, \
                                           name=layerName, trainable=True)
        if self.saveLayer:  self.layerOut[layerName] = x

        layerName = 'reshape' + layerIdx + self.detectorLayerID
        x = tfl.reshape(x, shape=(-1, finalLayerH, finalLayerW, \
                                   nAnchors, nClasses + 5), name=layerName)
        
        if self.saveLayer: self.layerOut[layerName] = x
        
        # Output size 15 x 20 x nAnchors x (nClasses + 5).
        
        return x
    
#===============================================================================

    def loss(self, logits, labels):
        '''
        Defines the region loss. This is the loss only for the second phase 
        of the training, when the network is only trained for detection with bbox.
        This network has already been trained earlier for classification.
        '''
        labels = tfl.convert_to_tensor(labels)
        labels = tfl.cast(labels, dtype=tfl.float32)    # Shape: (?, 14, 14, 5, 11)

        # The label is a finalLayerH x finalLayerW x nAnchors x (nClasses + 5) 
        # tensor. The 5 is for the anchor offsets (for x, y, w, h) and also a 
        # confidence factor which indicates if an object is there in that anchor
        # box or not. This value in the label is 1 if there is an object present.
        
        # Yolo puts different weights on the errors for the regions where there
        # are actual objects present and for regions where nothing is present.
        # So some kind of mask is needed to separate out these two kind of regions.
        # The confidence score mentioned above can be used to create this kind of
        # mask. For that we slice out this confidence channel from the label tensor.
        objectMask = labels[:, :, :, :, -1]   # The last channel is for confidence.
        objectMask = tfl.cast(objectMask, dtype=tfl.bool)   # Shape: (?, 14, 14, 5)
        #print(objectMask.get_shape())       
        
        # Now only taking the regions in the labels corresponding to the 1's in 
        # the mask. This will have only those parts of the label tensor which 
        # has the confidence as 1. These parts will be extracted into the new 
        # tensor. The new tensor will not have the other parts. So the shape of 
        # the output tensor is different from the input label tensor.
        maskedLabels = tfl.boolean_mask(labels, objectMask)    # Shape: (?, 11)
        #print(maskedLabels.get_shape())       
        
        # Now only taking the corresponding regions of the logits.
        maskedLogits = tfl.boolean_mask(logits, objectMask)    # Shape: (?, 11)

        # Now only taking the regions in the labels corresponding to the 0's in the mask.
        negMaskedLabels = tfl.boolean_mask(labels, tfl.logical_not(objectMask))

        # Now only taking the corresponding regions of the logits.
        negMaskedLogits = tfl.boolean_mask(logits, tfl.logical_not(objectMask))
        
#-------------------------------------------------------------------------------

        # Masked predictions of x, y, w and h offset.
        
        # Taking the sigmoid of the xy (eqn on page 3 of the yolo v2 paper)
        maskedLogitsXoffset = tfl.sigmoid(maskedLogits[:, -5])   # Shape: (?,)
        maskedLogitsYoffset = tfl.sigmoid(maskedLogits[:, -4])   # Shape: (?,)
        
        # Taking the exp of the wh (eqn on page 3 of the yolo v2 paper)
        maskedLogitsWoffset = maskedLogits[:, -3]   # Shape: (?,)
        maskedLogitsHoffset = maskedLogits[:, -2]   # Shape: (?,)

        # Taking the sigmoid of the object present prediction mask (this contributes 
        # to the 3rd term in the error function in page 4 of yolo '1' (not 2)).
        maskedLogitsConfidence = tfl.sigmoid(maskedLogits[:, -1])   # Shape: (?,)
        
        # Taking the sigmoid of the NO object present prediction mask (this contributes 
        # to the 4th term in the error function in page 4 of yolo '1' (not 2)).
        negMaskedLogitsConfidence = tfl.sigmoid(negMaskedLogits[:, -1])   # Shape: (?,)
        
        # Softmax for the one hot class prediction.
        maskedLogitsOneHotVec = tfl.nn.softmax(maskedLogits[:, : nClasses])   # Shape: (?, 6)
        #print(maskedLabelsOneHotVec.get_shape())
        
#-------------------------------------------------------------------------------

        # Masked labels of x, y, w and h offset.
        maskedLabelsXoffset = maskedLabels[:, -5]
        maskedLabelsYoffset = maskedLabels[:, -4]
        maskedLabelsWoffset = maskedLabels[:, -3]
        maskedLabelsHoffset = maskedLabels[:, -2]
        maskedLabelsConfidence = maskedLabels[:, -1]
        negMaskedLabelsConfidence = negMaskedLabels[:, -1]
        maskedLabelsOneHotVec = maskedLabels[:, : nClasses]
        
#-------------------------------------------------------------------------------

        # Calculating the losses.
        
        # Corresponds to 1st term of loss function in yolo '1' (not 2) paper (page 4).
        lossX = tfl.reduce_sum(tfl.square(maskedLabelsXoffset - maskedLogitsXoffset))
        lossY = tfl.reduce_sum(tfl.square(maskedLabelsYoffset - maskedLogitsYoffset))

        # Corresponds to 2nd term of loss function in yolo '1' (not 2) paper (page 4).
        lossW = tfl.reduce_sum(tfl.square(maskedLabelsWoffset - maskedLogitsWoffset))
        lossH = tfl.reduce_sum(tfl.square(maskedLabelsHoffset - maskedLogitsHoffset))
        
        # Corresponds to 3rd term of loss function in yolo '1' (not 2) paper (page 4).
        lossConfidence = tfl.reduce_sum(tfl.square(maskedLabelsConfidence - \
                                                   maskedLogitsConfidence))

        # Corresponds to 4th term of loss function in yolo '1' (not 2) paper (page 4).
        lossNegConfidence = tfl.reduce_sum(tfl.square(negMaskedLabelsConfidence - \
                                                      negMaskedLogitsConfidence))
        
        # Corresponds to 5th term of loss function in yolo '1' (not 2) paper (page 4).
        lossOneHotVec = tfl.reduce_sum(tfl.square(maskedLabelsOneHotVec - \
                                                  maskedLogitsOneHotVec))

#-------------------------------------------------------------------------------

        # Combining the losses.
        totalLoss = lambdaCoord * (lossX + lossY + lossW + lossH) + lossConfidence + \
                    lambdaNoObj * lossNegConfidence + lambdaClass * lossOneHotVec

#        # Loss per image.
#        meanLoss = totalLoss / (tfl.cast(tfl.shape(logits)[0], tfl.float32))

#        return meanLoss
                    
        return totalLoss

#===============================================================================

    def train(self, trainDir=None, validDir=None):
        '''
        Trains the model.
        '''
        if trainDir is None or validDir is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in train. Aborting.\n')
            sys.exit() 
        
        # SET INPUTS AND LABELS.
        
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches (especially the last batch).
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        # Labels are multi hot vectors.
        y = tfl.placeholder(dtype=tfl.float32, name='yPlaceholder', \
                            shape=[None, finalLayerH, finalLayerW, nAnchors, (nClasses + 5)])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)    # Model prediction logits.
            
            # List of model variables.
            listOfModelVars = [v for v in tfl.global_variables()]
            
            #for v in listOfModelVars:
                #print('Model: {}, Variable: {}'.format(modelName, v))

#-------------------------------------------------------------------------------

            # Modifying the predictions into proper formats.

            # The raw output of the model has to be converted into proper form 
            # by incorporating the sigmoid and softmax functions with them.
            predLogitsOneHotVec = tfl.nn.softmax(predLogits[:, :, :, :, : nClasses])
            predLogitsXoffset = tfl.sigmoid(predLogits[:, :, :, :, -5])
            predLogitsYoffset = tfl.sigmoid(predLogits[:, :, :, :, -4])
            predLogitsWoffset = tfl.exp(predLogits[:, :, :, :, -3])
            predLogitsHoffset = tfl.exp(predLogits[:, :, :, :, -2])    # Shape: (?, 14, 14, 5)
            
            # The anchor offsets needs to be multiplied with the anchor dimensions.
            # Creating some tensors of those anchor dimensions for this purpose.
            anchorWlist = [anc[0] for anc in anchorList]
            anchorWarr = np.tile(anchorWlist, (finalLayerH, finalLayerW, 1))            
            anchorWtensor = tfl.convert_to_tensor(anchorWarr, dtype=tfl.float32)
            anchorHlist = [anc[1] for anc in anchorList]
            anchorHarr = np.tile(anchorHlist, (finalLayerH, finalLayerW, 1))
            anchorHtensor = tfl.convert_to_tensor(anchorHarr, dtype=tfl.float32)
            
            predLogitsWoffset = tfl.multiply(predLogitsWoffset, anchorWtensor)
            predLogitsHoffset = tfl.multiply(predLogitsHoffset, anchorHtensor)
            predLogitsConfidence = tfl.sigmoid(predLogits[:, :, :, :, -1])

            # Now combining all these individual predlogits into one tensor.
            predLogitsXoffset = tfl.expand_dims(predLogitsXoffset, axis=-1)
            predLogitsYoffset = tfl.expand_dims(predLogitsYoffset, axis=-1)
            predLogitsWoffset = tfl.expand_dims(predLogitsWoffset, axis=-1)
            predLogitsHoffset = tfl.expand_dims(predLogitsHoffset, axis=-1)
            predLogitsConfidence = tfl.expand_dims(predLogitsConfidence, axis=-1)
            
            predResult = tfl.concat([predLogitsOneHotVec, predLogitsXoffset, \
                                      predLogitsYoffset, predLogitsWoffset, \
                                      predLogitsHoffset, predLogitsConfidence], axis=-1)

#-------------------------------------------------------------------------------
        
        # CALCULATE LOSS.
        
        loss = self.loss(logits=predLogits, labels=y)
        
        # DEFINE OPTIMIZER AND PERFORM BACKPROP.
        optimizer = self.optimizer
        
        # While executing an operation (such as trainStep), only the subgraph 
        # components relevant to trainStep will be executed. The 
        # update_moving_averages operation (for the batch normalization layers) 
        # is not a parent of trainStep in the computational graph, so it will 
        # never update the moving averages by default. To get around this, 
        # we have to explicitly tell the graph in the following manner.
        update_ops = tfl.get_collection(tfl.GraphKeys.UPDATE_OPS)
        with tfl.control_dependencies(update_ops):
            trainStep = optimizer.minimize(loss)

        # List of optimizer variables.
        listOfOptimizerVars = list(set(tfl.global_variables()) - set(listOfModelVars))

        #for v in listOfOptimizerVars:
            #print('Optimizer: {}, Variable: {}'.format(self.optimizerName, v))

#-------------------------------------------------------------------------------
        
        # CREATE A LISTS TO HOLD LOSS VALUES.
        
        # This list will have strings for each epoch. Each of these strings 
        # will be like the following:
        # 'epoch, learningRate, trainLoss, validLoss'
        statistics = []
        
        # Format of the statistics values.
        statisticsFormat = 'epoch, learningRate, batchSize, trainLoss, validLoss, ' \
                           'epochProcessTime'
                        
#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        print('\nStarting session. Optimizer: {}, Learning Rate: {}, ' \
               'Batch Size: {}'.format(self.optimizerName, learningRate, batchSize))
        
        self.isTraining = True    # Enabling the training flag. 
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPathDetector, \
                                                              training=self.isTraining)
        startEpoch = latestEpoch + 1    # Start from the next epoch.

#-------------------------------------------------------------------------------
        
        # LOAD PARAMETERS FROM CHECKPOINTS.
        
        if ckptPath != None:    # Only when some checkpoint is found.
            with open(jsonFilePath, 'r') as infoFile:
                infoDict = json.load(infoFile)
                
            if infoDict['learningRate'] == learningRate and infoDict['batchSize'] == batchSize:
                # Since all old variables will be loaded here, so we do not need
                # to initialize any other variables.
                
                # Now define a new saver with all the variables.
                saver = tfl.train.Saver(listOfModelVars + listOfOptimizerVars)
                saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.
                saver.restore(sess, ckptPath)

                print('\nReloaded ALL variables from checkpoint: {}\n'.format(\
                                                            ckptPath))

#-------------------------------------------------------------------------------

            else:
                # else load only weight and biases and skip optimizer 
                # variables. Otherwise there will be errors.
                # So define the saver with the list of only model variables.
                saver = tfl.train.Saver(listOfModelVars)
                saver.restore(sess, ckptPath)
                # Since only the model variables will be loaded here, so we
                # have to initialize other variables (optimizer variables)
                # separately.
                sess.run(tfl.variables_initializer(listOfOptimizerVars))

                # The previous saver will only save the listOfModelVars
                # as it is defined using only those variables (as the 
                # checkpoint can only give us those values as valid for
                # restoration). But now since we have all the varibles 
                # loaded and all the new ones initialized, so we redefine 
                # the saver to include all the variables. So that while 
                # saving in the end of the epoch, it can save all the 
                # variables (and not just the listOfModelVars).
                saver = tfl.train.Saver(listOfModelVars + listOfOptimizerVars)
                saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.

                # Load mean and std values.
                mean, std = infoDict['mean'], infoDict['std']
                
                print('\nCurrent parameters:\nlearningRate: {}, batchSize: {}' \
                       '\nPrevious parameters (inside checkpoint {}):\nlearningRate: ' \
                       '{}, batchSize: {}\nThey are different.\nSo reloaded only ' \
                       'MODEL variables from checkpoint: {}\nAnd initialized ' \
                       'other variables.\n'.format(learningRate, batchSize, ckptPath, \
                       infoDict['learningRate'], infoDict['batchSize'], ckptPath))

            # Reloading accuracy and loss statistics, mean and std from checkpoint.
            statistics = infoDict['statistics']
            mean = np.array(infoDict['mean'])
            std = np.array(infoDict['std'])
            minValidLoss = infoDict['minValidLoss']

            # If the batchsize changes, then the minValidLoss should also be 
            # scaled according to that.
            if batchSize != infoDict['batchSize']:
                minValidLoss = infoDict['minValidLoss'] * batchSize / infoDict['batchSize']

#-------------------------------------------------------------------------------

        else:
            # When there is no saved checkpoints then load the weights of the 
            # layers which are common between the detector and the classifier
            # from the latest classifier checkpoint and initialize the rest of 
            # the variables which are specific to the new layers added for the 
            # classifier phase.

            # List of classifier variables.
            listOfClassifierVars = []
            for v in tfl.global_variables():
                # Include only those variables whose name has this model's name in it
                # and does not contain the self.detectorLayerID in them.
                if v.name.find(modelName) >= 0 and v.name.find(self.detectorLayerID) < 0:
                    listOfClassifierVars.append(v)
                    #print('Classifier: {}, Variable: {}'.format(modelName, v))

            # Finding latest checkpoint of the classifier (not detector) and 
            # loading the variables of the initial layers (which are common 
            # between the classifier and the detector) into this network.
            _, ckptPathClassifier, _ = findLatestCkpt(ckptDirPath, training=self.isTraining)
            
            if ckptPathClassifier == None:
                print('\nERROR: No saved checkpoint for the classification found. ' \
                       'So cannot initialize the layers of the detector. Aborting.\n')
                sys.exit()

#-------------------------------------------------------------------------------
                
            # Loading these variable weights from the classifier checkpoint using
            # a saver that is defined only for the classifier variables.
            saver = tfl.train.Saver(listOfClassifierVars)
            saver.restore(sess, ckptPathClassifier)
            
            # Now initialize the rest of the layers and the optimizer variables.
            listOfOtherVars = list(set(tfl.global_variables()) - set(listOfClassifierVars))
                        
            sess.run(tfl.variables_initializer(listOfOtherVars))

            # The previous saver will only save the listOfClassifierVars
            # as it is defined using only those variables (as the 
            # checkpoint can only give us those values as valid for
            # restoration). But now since we have all the varibles 
            # loaded and all the new ones initialized, so we redefine 
            # the saver to include all the variables. So that while 
            # saving in the end of the epoch, it can save all the 
            # variables (and not just the listOfClassifierVars).
            saver = tfl.train.Saver(listOfClassifierVars + listOfOtherVars)
            saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.

            print('\nReloaded the common variables from the classifier ' \
                     'checkpoint: {}\n'.format(ckptPathClassifier))
            
            # Calculate mean and std.
            if recordedMean is None or recordedStd is None:
                mean, std = datasetMeanStd(trainDir)
                print('\nMean of {}: {}'.format(trainDir, mean))
                print('\nStd of {}: {}'.format(trainDir, std))
            else:
                mean, std = recordedMean, recordedStd

            minValidLoss = np.inf

#-------------------------------------------------------------------------------
        
        # TRAINING AND VALIDATION.
        
        print('\nStarted Training...\n')
        
        for epoch in range(startEpoch, nEpochs+1):
            # epoch will be numbered from 1 to 150 if there are 150 epochs.
            # Is is not numbered from 0 to 149.

            epochProcessTime = time.time()
            
            # TRAINING PHASE.
            
            self.isTraining = True    # Enabling training flag at the start of 
            # the training phase of each epoch as it will be disabled in the 
            # corresponding validaton phase. 

            # This list contains the filepaths for all the images in trainDir.
            # If there are unwanted files like '.thumbs' etc. then those are 
            # filtered as well.
            listOfRemainingTrainImg = [i for i in os.listdir(\
                                        os.path.join(trainDir, 'images')) \
                                                if i[-3:] == 'png']
        
            nTrainImgs = len(listOfRemainingTrainImg)
            nTrainBatches = int(np.ceil(nTrainImgs / batchSize))
            
            # Interval at which the status will be printed in the terminal.
            printInterval = 1 if nTrainBatches < 10 else int(nTrainBatches / 10)
            
            trainLoss = 0.0
            trainBatchIdx = 0    # Counts the number of batch processed.
            
            # Storing information of current epoch for recording later in statistics.
            # This is recorded as a string for better visibility in the json file.
            currentStatistics = '{}, {}, {}, '.format(epoch, learningRate, batchSize)
            
#-------------------------------------------------------------------------------
            
            # Scan entire training dataset.
            while len(listOfRemainingTrainImg) > 0:

                trainBatchProcessTime = time.time()

                # Creating batches. Also listOfRemainingTrainImg is updated.
                trainImgBatch, trainLabelBatch, _, _, listOfRemainingTrainImg, _ = \
                    createBatchForDetection(trainDir, listOfRemainingTrainImg, \
                                                  batchSize, shuffle=True)

                feedDict = {x: trainImgBatch, y: trainLabelBatch}
                if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                    trainLayerOut = sess.run(self.layerOut, feed_dict=feedDict)

                trainPredLogits = sess.run(predLogits, feed_dict=feedDict)
                trainBatchLoss = sess.run(loss, feed_dict=feedDict)
                sess.run(trainStep, feed_dict=feedDict)
                
                # Evaluating the prediction result tensor.
                trainPredResult = sess.run(predResult, feed_dict=feedDict)
                
                trainLoss += (trainBatchLoss / nTrainBatches)
                
#                print(trainPredLogits)

#-------------------------------------------------------------------------------                

                trainBatchIdx += 1
                trainBatchProcessTime = time.time() - trainBatchProcessTime

                # Print the status on the terminal every 10 batches.
                if trainBatchIdx % printInterval == 0:
                    print('Epoch: {}/{},\tBatch: {}/{},\tBatch loss: ' \
                           '{:0.6f},\tProcess time for {} batch: {}'.format(epoch, \
                           nEpochs, trainBatchIdx, nTrainBatches, trainBatchLoss, \
                           printInterval, prettyTime(trainBatchProcessTime*printInterval)))
                
            # Recording training loss and accuracy in current statistics string.
            currentStatistics += '{}, '.format(trainLoss)
            
#-------------------------------------------------------------------------------
            
            # VALIDATION PHASE.
            
            self.isTraining = False    # Disabling the training flag.
            
            # This list contains the filepaths for all the images in validDir.
            # If there are unwanted files like '.thumbs' etc. then those are 
            # filtered as well.
            listOfRemainingValidImg = [i for i in os.listdir(\
                                        os.path.join(validDir, 'images')) \
                                                if i[-3:] == 'png']
            
            nValidImgs = len(listOfRemainingValidImg)
            nValidBatches = int(np.ceil(nValidImgs / batchSize))
            
            # Interval at which the status will be printed in the terminal.
            printInterval = 1 if nValidBatches < 3 else int(nValidBatches / 3)

            validLoss = 0.0
            validBatchIdx = 0    # Counts the number of batch processed.
            
            print('\n\nValidation phase for epoch {}.\n'.format(epoch))
            
#-------------------------------------------------------------------------------
            
            # Scan entire validation dataset.
            while len(listOfRemainingValidImg) > 0:
                
                validBatchProcessTime = time.time()

                # Creating batches. Also listOfRemainingValidImg is updated.
                # The shuffle is off for validation and the mean and std are
                # the same as calculated on the training set.
                validImgBatch, validLabelBatch, _, _, listOfRemainingValidImg, _ = \
                    createBatchForDetection(validDir, listOfRemainingValidImg, \
                                                  batchSize, shuffle=False)
                    
                feedDict = {x: validImgBatch, y: validLabelBatch}
                if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                    validLayerOut = sess.run(self.layerOut, feed_dict=feedDict)
                
                validPredLogits = sess.run(predLogits, feed_dict=feedDict)
                validBatchLoss = sess.run(loss, feed_dict=feedDict)
                
                # Evaluating the prediction result tensor.
                validPredResult = sess.run(predResult, feed_dict=feedDict)

                validLoss += (validBatchLoss / nValidBatches)
                                    
#------------------------------------------------------------------------------- 

                validBatchIdx += 1
                validBatchProcessTime = time.time() - validBatchProcessTime

                # Print the status on the terminal every 10 batches.
                if validBatchIdx % printInterval == 0:     
                    print('Epoch: {}/{},\tBatch: {}/{},\tBatch loss: ' \
                           '{:0.6f},\tProcess time for {} batch: {}'.format(epoch, \
                           nEpochs, validBatchIdx, nValidBatches, validBatchLoss, \
                           printInterval, prettyTime(validBatchProcessTime*printInterval)))
            
            # Recording validation accuracy in current statistics string.
            currentStatistics += '{}, '.format(validLoss)

#-------------------------------------------------------------------------------
            
            # STATUS UPDATE.
                        
            epochProcessTime = time.time() - epochProcessTime

            # Recording epoch processing time in current statistics string.
            currentStatistics += '{}'.format(prettyTime(epochProcessTime))
            
            # Noting accuracy after the end of all batches.
            statistics.append(currentStatistics)
            
            print('\nEpoch {} done. Epoch time: {}, Train loss: {:0.6f}, ' \
                   'Valid loss: {:0.6f}\n'.format(epoch, \
                   prettyTime(epochProcessTime), trainLoss, validLoss))

            # Saving the variables at some intervals, only if there is 
            # improvement in validation accuracy.
            if epoch % modelSaveInterval == 0 and validLoss < minValidLoss:
                ckptSavePath = os.path.join(ckptDirPathDetector, savedCkptNameDetector)
                saver.save(sess, save_path=ckptSavePath, global_step=epoch)
                
                minValidLoss = validLoss    # Updating the minValidLoss.
                
                # Saving the important info like learning rate, batch size,
                # and training error for the current epoch in a json file.
                # Converting the mean and std into lists before storing as
                # json cannot store numpy arrays. And also saving the training
                # and validation loss statistics.
                jsonInfoFilePath = ckptSavePath + '-' + str(epoch) + '.json'
                with open(jsonInfoFilePath, 'w') as infoFile:
                    infoDict = {'epoch': epoch, 'batchSize': batchSize, \
                                 'learningRate': learningRate, \
                                 'mean': list(mean), 'std': list(std), \
                                 'minValidLoss': minValidLoss, \
                                 'statisticsFormat': statisticsFormat, \
                                 'statistics': statistics}
                    
                    json.dump(infoDict, infoFile, sort_keys=False, \
                               indent=4, separators=(',', ': '))
                
                print('Checkpoint saved.\n')
            
            # Updating the maxValidAcc value.
            elif validLoss < minValidLoss:
                minValidLoss = validLoss
            
#-------------------------------------------------------------------------------
        
        self.isTraining = False   # Indicates the end of training.
        print('\nTraining completed with {} epochs.'.format(nEpochs))

        sess.close()        # Closing the session.
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.

#===============================================================================

    def test(self, testDir=None):
        '''
        Tests the model.
        '''
        if testDir is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in test. Aborting.\n')
            sys.exit() 

        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS AND LABELS.
        
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches (especially the last batch).
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        # Labels are multi hot vectors.
        y = tfl.placeholder(dtype=tfl.float32, name='yPlaceholder', \
                            shape=[None, finalLayerH, finalLayerW, nAnchors, (nClasses + 5)])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.

            # List of model variables.
            listOfModelVars = []
            for v in tfl.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find(modelName) >= 0:
                    listOfModelVars.append(v)
                    #print('Model: {}, Variable: {}'.format(modelName, v))
                    
#-------------------------------------------------------------------------------

            # Modifying the predictions into proper formats.

            # The raw output of the model has to be converted into proper form 
            # by incorporating the sigmoid and softmax functions with them.
            predLogitsOneHotVec = tfl.nn.softmax(predLogits[:, :, :, :, : nClasses])
            predLogitsXoffset = tfl.sigmoid(predLogits[:, :, :, :, -5])
            predLogitsYoffset = tfl.sigmoid(predLogits[:, :, :, :, -4])
            predLogitsWoffset = tfl.exp(predLogits[:, :, :, :, -3])
            predLogitsHoffset = tfl.exp(predLogits[:, :, :, :, -2])
            
            # The anchor offsets needs to be multiplied with the anchor dimensions.
            # Creating some tensors of those anchor dimensions for this purpose.
            anchorWlist = [anc[0] for anc in anchorList]
            anchorWarr = np.tile(anchorWlist, (finalLayerH, finalLayerW, 1))            
            anchorWtensor = tfl.convert_to_tensor(anchorWarr, dtype=tfl.float32)
            anchorHlist = [anc[1] for anc in anchorList]
            anchorHarr = np.tile(anchorHlist, (finalLayerH, finalLayerW, 1))
            anchorHtensor = tfl.convert_to_tensor(anchorHarr, dtype=tfl.float32)
            
            predLogitsWoffset = tfl.multiply(predLogitsWoffset, anchorWtensor)
            predLogitsHoffset = tfl.multiply(predLogitsHoffset, anchorHtensor)
            predLogitsConfidence = tfl.sigmoid(predLogits[:, :, :, :, -1])

            # Now combining all these individual predlogits into one tensor.
            predLogitsXoffset = tfl.expand_dims(predLogitsXoffset, axis=-1)
            predLogitsYoffset = tfl.expand_dims(predLogitsYoffset, axis=-1)
            predLogitsWoffset = tfl.expand_dims(predLogitsWoffset, axis=-1)
            predLogitsHoffset = tfl.expand_dims(predLogitsHoffset, axis=-1)
            predLogitsConfidence = tfl.expand_dims(predLogitsConfidence, axis=-1)
            
            predResult = tfl.concat([predLogitsOneHotVec, predLogitsXoffset, \
                                      predLogitsYoffset, predLogitsWoffset, \
                                      predLogitsHoffset, predLogitsConfidence], axis=-1)

#-------------------------------------------------------------------------------

        # CALCULATE LOSS.
        
        loss = self.loss(logits=predLogits, labels=y)

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPathDetector, \
                                                              training=self.isTraining)

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tfl.train.Saver(listOfModelVars)
            saver.restore(sess, ckptPath)            
            print('\nReloaded ALL variables from checkpoint: {}'.format(\
                                                        ckptPath))
        else:
            # When there are no valid checkpoints.
            print('\nNo valid checkpoints found. Aborting.\n')
            return

        with open(jsonFilePath, 'r') as infoFile:
            infoDict = json.load(infoFile)

        # Reloading mean and std from checkpoint.
        mean = np.array(infoDict['mean'])
        std = np.array(infoDict['std'])

#-------------------------------------------------------------------------------
                    
        print('\nStarted Testing...\n')
            
        # TESTING PHASE.

        testingTime = time.time()

        # This list contains the filepaths for all the images in validDir.
        # If there are unwanted files like '.thumbs' etc. then those are 
        # filtered as well.
        listOfRemainingTestImg = [i for i in os.listdir(\
                                  os.path.join(testDir, 'images')) \
                                            if i[-3:] == 'png']
        
        nTestImgs = len(listOfRemainingTestImg)        
        nTestBatches = int(np.ceil(nTestImgs / batchSize))
        
        # Interval at which the status will be printed in the terminal.
        printInterval = 1 if nTestBatches < 3 else int(nTestBatches / 3)
        
        testLoss, testAccClassification = 0.0, 0.0
        testBatchIdx = 0    # Counts the number of batches processed.

#-------------------------------------------------------------------------------
        
        # These lists will record the predicted and ground truth boxes for all
        # the images in the entire test dataset. These records will be used to 
        # calculate the mAP of the dataset.
        allTestMultiHot, allTestClassIdxAndBbox, allDetectedClassScores, \
                    allDetectedBboxes, allDetectedClassIdxes = [], [], [], [], []
        
        # Just like the totalClsInstances, this totalDetectedClsInstances indicates
        # the number of times the class objects has been detected properly in the
        # images. This two will be later used to identify the individual class
        # detection accuracy.
        totalDetectedClsInstances = np.zeros(nClasses)

#-------------------------------------------------------------------------------
            
        # Scan entire validation dataset.
        while len(listOfRemainingTestImg) > 0:
        
            testBatchProcessTime = time.time()

            # Creating batches. Also listOfRemainingValidImg is updated.
            # The shuffle is off for validation and the mean and std are
            # the same as calculated on the training set.
            testImgBatch, testLabelBatch, testLabelBatchMultiHot, \
            testLabelBatchClassIdxAndBbox, listOfRemainingTestImg, \
            listOfSelectedTestImg = createBatchForDetection(testDir, \
                                        listOfRemainingTestImg, batchSize, shuffle=False)
            
            feedDict = {x: testImgBatch, y: testLabelBatch}
            if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                testLayerOut = sess.run(self.layerOut, feed_dict=feedDict)
            
            testPredLogits = sess.run(predLogits, feed_dict=feedDict)
            testBatchLoss = sess.run(loss, feed_dict=feedDict)
            
            # Evaluating the prediction result tensor.
            testPredResult = sess.run(predResult, feed_dict=feedDict)
            
            testLoss += (testBatchLoss / nTestBatches)
                                                            
#------------------------------------------------------------------------------- 

            # Evaluating the classification accuracy in this detection phase.
            # This is just to check if the classification accuracy is still 
            # comparable to what was obtained in the classification phase only.
            detectedBatchClassScores, detectedBatchClassIdxes, detectedBatchClassNames, \
                            detectedBatchBboxes = nonMaxSuppression(testPredResult)
            
            # Converting to list, so detectedBatchClassIdxes is a list of lists now.
            detectedBatchClassIdxes = detectedBatchClassIdxes.tolist()

            # Creating the multihot vector label batch to check the accuracy.
            predLabelMultiHot = np.zeros(testLabelBatchMultiHot.shape, dtype=np.int32)
            
            for idx, i in enumerate(detectedBatchClassIdxes):
                multiHotLabel = np.zeros(nClasses, dtype=np.int32)  # Blank label.
                
                # Make the positions (corresponding to the indexes listed in i) 1.
                for j in i:     multiHotLabel[j] = 1
                
                predLabelMultiHot[idx] = multiHotLabel
                
#------------------------------------------------------------------------------- 
            
            matches = np.array(predLabelMultiHot == testLabelBatchMultiHot, dtype=np.int32)
            
            # Counting the number of class instances that were predicted properly 
            # in the current batch.
            nDetectedClsInstances = np.sum(matches, axis=0)
            totalDetectedClsInstances += nDetectedClsInstances
            
            # The matches will be an array of batchSize x 5 (if there are 5 classes).
            # Since independent probabilities are calculated, so there may be
            # more than one 1's or 0's among these 5 elements.
            # Hence, a completely correct prediction will have a True match 
            # for all of these 5 elements. The sum of this matches array is
            # calculated along the columns. If that results in 5 then it 
            # means that a perfect match has happened.
            matches = np.sum(matches, axis=1)
            perfectMatch = np.asarray(np.ones(matches.shape) * nClasses, dtype=np.int32)
            matches1 = np.array(matches == perfectMatch, dtype=np.int32)

            testAccClassification += (100*np.sum(matches1)) / nTestImgs
            
#-------------------------------------------------------------------------------
            
            # The testLabelBatchClassIdxAndBbox is a list of list of list.
            # Each of the element of this list is an entry for 1 image in the 
            # batch. Each of these entries is a list containing as many entries
            # as the number of objects present in the image. Each of these last 
            # tier of lists has 5 elements, each of one object. They have the 
            # class idx, center x, center y, width and height of the bounding
            # box of the object.
            allTestMultiHot += testLabelBatchMultiHot.tolist()
            allTestClassIdxAndBbox += testLabelBatchClassIdxAndBbox
            allDetectedClassScores += detectedBatchClassScores.tolist()
            allDetectedBboxes += detectedBatchBboxes.tolist()
            allDetectedClassIdxes += detectedBatchClassIdxes
            # No need to convert the last one into list as it is already converted
            # into a list earlier after the function nonMaxSuppression was called.
            
#-------------------------------------------------------------------------------

            testBatchIdx += 1
            testBatchProcessTime = time.time() - testBatchProcessTime

            # Printing current status of testing.
            # print the status on the terminal every 10 batches.
            if testBatchIdx % printInterval == 0:
                print('Batch: {}/{},\tBatch loss: {:0.6f},\tProcess time for {} ' \
                       'batch: {}'.format(testBatchIdx, nTestBatches, testBatchLoss, \
                        printInterval, prettyTime(testBatchProcessTime*printInterval)))

#-------------------------------------------------------------------------------

            # Creating a list of images which have been misclassified.            
            if testBatchIdx == 1:
                misclassificationList = [m for mdx, m in enumerate(\
                                          listOfSelectedTestImg) if matches1[mdx] == 0]
            else:
                misclassificationList += [m for mdx, m in enumerate(\
                                           listOfSelectedTestImg) if matches1[mdx] == 0]

#-------------------------------------------------------------------------------
        
        # Calculate the mean average precision.
        mAP, APlist = calculateMAP(allTestMultiHot, allTestClassIdxAndBbox, \
                                    allDetectedClassIdxes, allDetectedClassScores, \
                                    allDetectedBboxes)

        # Calculate the individual class prediction accuracy.
        clsDetectionAcc = (totalDetectedClsInstances / nTestImgs) * 100

#-------------------------------------------------------------------------------

        # Now calculating the total number of true and false positives for each 
        # class over the entire dataset.
        
        # There will be 3 lists. One will have the number of actual positive
        # instance for each class over the entire datasets. The other will have
        # the number of true positive instance for each class over the entire
        # dataset and the third will have the number of false positive instance
        # for each class over the entire dataset.
        actualPos = np.zeros(nClasses, dtype=np.int)
        predPos = np.zeros(nClasses, dtype=np.int)
        truePos = np.zeros(nClasses, dtype=np.int)
        falsePos = np.zeros(nClasses, dtype=np.int)
        errorInCount = np.zeros(nClasses, dtype=np.int)
        
        for idx, i in enumerate(allTestClassIdxAndBbox):
            actualPosInCurrentImg = np.zeros(nClasses, dtype=np.int)
            truePosInCurrentImg = np.zeros(nClasses, dtype=np.int)
            falsePosInCurrentImg = np.zeros(nClasses, dtype=np.int)
            predPosInCurrentImg = np.zeros(nClasses, dtype=np.int)
            
            # Calculating the actual positives for the current image.
            for currentObj in i:
                currentObjIdx = currentObj[0]
                actualPosInCurrentImg[currentObjIdx] += 1
                
            # Calculating the predicted positives for the current image.
            for predObjIdx in allDetectedClassIdxes[idx]:
                predPosInCurrentImg[predObjIdx] += 1
            
            # Calculating the true and false positives for the current image.
            for index, (a, p) in enumerate(zip(actualPosInCurrentImg, predPosInCurrentImg)):
                if a > p:   # Number of actual +ves > number of predicted +ves.
                    truePosInCurrentImg[index] = p
                    falsePosInCurrentImg[index] = 0
                elif a < p: # Number of actual +ves < number of predicted +ves.
                    truePosInCurrentImg[index] = a
                    falsePosInCurrentImg[index] = p - a
                else:   # Number of actual +ves == number of predicted +ves.
                    truePosInCurrentImg[index] = p
                    falsePosInCurrentImg[index] = 0
                    
#            print(idx, actualPosInCurrentImg, predPosInCurrentImg)

            # Now calculating these metrics over the complete dataset.
            actualPos += actualPosInCurrentImg
            predPos += predPosInCurrentImg
            truePos += truePosInCurrentImg
            falsePos += falsePosInCurrentImg
            errorInCount += np.abs(actualPosInCurrentImg - predPosInCurrentImg)
            
        # Now calculating the precision and reacall of the individual classes 
        # over the entire dataset.
        recall = truePos / actualPos
        precision = truePos / predPos
        f1score = 2 * (precision * recall) / (precision + recall)
        meanErrorInCount = errorInCount / actualPos
        
#-------------------------------------------------------------------------------
                
        # Printing the average precision and detection accuracy of individual classes
        # and also the recall (which gives an estimate of the error in count 
        # (Ncorrect / Ntotal).
        print('\n')
        for cdx, AP in enumerate(APlist):
            print('Average Precision (AP) for class {}: {:0.3f} % ; ' \
                   'Detection Accuracy: {:0.3f} % ; \nNcorrect / Ntotal (Recall): {:0.3f} ; ' \
                   'Ncorrect / Npredicted (Precision): {:0.3f} ; F1score: {:0.3f} ; ' \
                   'Mean Error in Count: {:0.3f}'.format(classIdxToName[cdx], AP, \
                        clsDetectionAcc[cdx], recall[cdx], precision[cdx], f1score[cdx], \
                        meanErrorInCount[cdx]))
        
        print('\nMean Average Precision (mAP) over given dataset: {:0.3f} %'.format(mAP))

#-------------------------------------------------------------------------------

        # Save the misclassified examples list in a json file (use json.dumps 
        # not json.dump).
        with open('misclassified_example_list_detection_{}.json'.format(testDir), 'w') \
                                                                                as infoFile:
            infoDict = misclassificationList
            json.dump(misclassificationList, infoFile, indent=4, separators=(',', ':'))

#-------------------------------------------------------------------------------

        testingTime = time.time() - testingTime
        print('\n\nTesting done. Test Loss: {:0.6f}, ' \
               'Test Classification Accuracy: {:0.3f} %, ' \
               'Testing time: {}'.format(testLoss, testAccClassification, \
                                          prettyTime(testingTime)))

        sess.close()        # Closing the session.
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.
        
        return testLoss, testingTime

#===============================================================================

    def batchInference(self, imgBatch):
        '''
        This function evaluates the output of the model on an unknown batch of
        images (4d numpy array) and returns the predicted labels as a batch as well.
        '''
        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS.
        
        b, h, w, _ = imgBatch.shape

        # All the images in the batch are already resized to the appropriate shape 
        # when the batch was created, hence no need to resize again.
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.

            # List of model variables.
            listOfModelVars = []
            for v in tfl.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find(modelName) >= 0:
                    listOfModelVars.append(v)
                    #print('Model: {}, Variable: {}'.format(modelName, v))

#-------------------------------------------------------------------------------

            # Modifying the predictions into proper formats.

            # The raw output of the model has to be converted into proper form 
            # by incorporating the sigmoid and softmax functions with them.
            predLogitsOneHotVec = tfl.nn.softmax(predLogits[:, :, :, :, : nClasses])
            predLogitsXoffset = tfl.sigmoid(predLogits[:, :, :, :, -5])
            predLogitsYoffset = tfl.sigmoid(predLogits[:, :, :, :, -4])
            predLogitsWoffset = tfl.exp(predLogits[:, :, :, :, -3])
            predLogitsHoffset = tfl.exp(predLogits[:, :, :, :, -2])
           
            # The anchor offsets needs to be multiplied with the anchor dimensions.
            # Creating some tensors of those anchor dimensions for this purpose.
            anchorWlist = [anc[0] for anc in anchorList]
            anchorWarr = np.tile(anchorWlist, (finalLayerH, finalLayerW, 1))            
            anchorWtensor = tfl.convert_to_tensor(anchorWarr, dtype=tfl.float32)
            anchorHlist = [anc[1] for anc in anchorList]
            anchorHarr = np.tile(anchorHlist, (finalLayerH, finalLayerW, 1))
            anchorHtensor = tfl.convert_to_tensor(anchorHarr, dtype=tfl.float32)

            predLogitsWoffset = tfl.multiply(predLogitsWoffset, anchorWtensor)
            predLogitsHoffset = tfl.multiply(predLogitsHoffset, anchorHtensor)
            predLogitsConfidence = tfl.sigmoid(predLogits[:, :, :, :, -1])

            # Now combining all these individual predlogits into one tensor.
            predLogitsXoffset = tfl.expand_dims(predLogitsXoffset, axis=-1)
            predLogitsYoffset = tfl.expand_dims(predLogitsYoffset, axis=-1)
            predLogitsWoffset = tfl.expand_dims(predLogitsWoffset, axis=-1)
            predLogitsHoffset = tfl.expand_dims(predLogitsHoffset, axis=-1)
            predLogitsConfidence = tfl.expand_dims(predLogitsConfidence, axis=-1)
            
            predResult = tfl.concat([predLogitsOneHotVec, predLogitsXoffset, \
                                      predLogitsYoffset, predLogitsWoffset, \
                                      predLogitsHoffset, predLogitsConfidence], axis=-1)

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPathDetector, \
                                                              training=self.isTraining)

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tfl.train.Saver(listOfModelVars)
            saver.restore(sess, ckptPath)            
#            print('\nReloaded ALL variables from checkpoint: {}'.format(\
#                                                        ckptPath))
        else:
            # When there are no valid checkpoints.
            print('\nNo valid checkpoints found. Aborting.\n')
            return

        with open(jsonFilePath, 'r') as infoFile:
            infoDict = json.load(infoFile)
                
        # Reloading mean and std from checkpoint.
        mean = np.array(infoDict['mean'])
        std = np.array(infoDict['std'])

#-------------------------------------------------------------------------------

#        # Normalizing by mean and std as done in case of training.
#        imgBatch = (imgBatch - mean) / std

        # Converting image to range 0 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        imgBatch = np.asarray(imgBatch, dtype=np.float32) / 127.5 - 1.0
        
        feedDict = {x: imgBatch}
        if self.saveLayer:    # Evaluate layer outputs if this flag is True.
            inferLayerOut = sess.run(self.layerOut, feed_dict=feedDict)
        else:   inferLayerOut = None

        inferPredLogits = sess.run(predLogits, feed_dict=feedDict)
        
        inferPredResult = sess.run(predResult, feed_dict=feedDict)
                                                                                        
#-------------------------------------------------------------------------------
        
        sess.close()
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.
        
        return inferLayerOut, inferPredLogits, inferPredResult, mean, std
    
#===============================================================================

    def runInLoop(self, inferDir):
        '''
        This function evaluates the output of the model on an unknown set of images
        located inside the inferDir. The inference on these images are done in a loop.
        '''
        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS.
        
        # All the images in the batch are already resized to the appropriate shape 
        # when the batch was created, hence no need to resize again.
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.

            # List of model variables.
            listOfModelVars = []
            for v in tfl.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find(modelName) >= 0:
                    listOfModelVars.append(v)
                    #print('Model: {}, Variable: {}'.format(modelName, v))

#-------------------------------------------------------------------------------

            # Modifying the predictions into proper formats.

            # The raw output of the model has to be converted into proper form 
            # by incorporating the sigmoid and softmax functions with them.
            predLogitsOneHotVec = tfl.nn.softmax(predLogits[:, :, :, :, : nClasses])
            predLogitsXoffset = tfl.sigmoid(predLogits[:, :, :, :, -5])
            predLogitsYoffset = tfl.sigmoid(predLogits[:, :, :, :, -4])
            predLogitsWoffset = tfl.exp(predLogits[:, :, :, :, -3])
            predLogitsHoffset = tfl.exp(predLogits[:, :, :, :, -2])
           
            # The anchor offsets needs to be multiplied with the anchor dimensions.
            # Creating some tensors of those anchor dimensions for this purpose.
            anchorWlist = [anc[0] for anc in anchorList]
            anchorWarr = np.tile(anchorWlist, (finalLayerH, finalLayerW, 1))            
            anchorWtensor = tfl.convert_to_tensor(anchorWarr, dtype=tfl.float32)
            anchorHlist = [anc[1] for anc in anchorList]
            anchorHarr = np.tile(anchorHlist, (finalLayerH, finalLayerW, 1))
            anchorHtensor = tfl.convert_to_tensor(anchorHarr, dtype=tfl.float32)

            predLogitsWoffset = tfl.multiply(predLogitsWoffset, anchorWtensor)
            predLogitsHoffset = tfl.multiply(predLogitsHoffset, anchorHtensor)
            predLogitsConfidence = tfl.sigmoid(predLogits[:, :, :, :, -1])

            # Now combining all these individual predlogits into one tensor.
            predLogitsXoffset = tfl.expand_dims(predLogitsXoffset, axis=-1)
            predLogitsYoffset = tfl.expand_dims(predLogitsYoffset, axis=-1)
            predLogitsWoffset = tfl.expand_dims(predLogitsWoffset, axis=-1)
            predLogitsHoffset = tfl.expand_dims(predLogitsHoffset, axis=-1)
            predLogitsConfidence = tfl.expand_dims(predLogitsConfidence, axis=-1)
            
            predResult = tfl.concat([predLogitsOneHotVec, predLogitsXoffset, \
                                      predLogitsYoffset, predLogitsWoffset, \
                                      predLogitsHoffset, predLogitsConfidence], axis=-1)

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPathDetector, \
                                                              training=self.isTraining)

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tfl.train.Saver(listOfModelVars)
            saver.restore(sess, ckptPath)            
#            print('\nReloaded ALL variables from checkpoint: {}'.format(\
#                                                        ckptPath))
        else:
            # When there are no valid checkpoints.
            print('\nNo valid checkpoints found. Aborting.\n')
            return

        with open(jsonFilePath, 'r') as infoFile:
            infoDict = json.load(infoFile)
                
        # Reloading mean and std from checkpoint.
        mean = np.array(infoDict['mean'])
        std = np.array(infoDict['std'])

#-------------------------------------------------------------------------------

        # Running the loop now.
        listOfImgs = os.listdir(os.path.join(inferDir, 'images'))
        nImgs = len(listOfImgs)
        
        for i in range(nImgs):
            startTime = time.time()
            
            img = cv2.imread(os.path.join(inferDir, 'images', listOfImgs[i]))
            imgBatch = np.array([img])

#           # Normalizing by mean and std as done in case of training.
#           imgBatch = (imgBatch - mean) / std

            # Converting image to range 0 to 1.
            # The image is explicitly converted to float32 to match the type 
            # specified in the placeholder. If img would have been directly divided
            # by 127.5, then it would result in np.float64.
            imgBatch = np.asarray(imgBatch, dtype=np.float32) / 127.5 - 1.0
        
            feedDict = {x: imgBatch}

            inferPredLogits = sess.run(predLogits, feed_dict=feedDict)
            
            inferPredResult = sess.run(predResult, feed_dict=feedDict)
            
            frameRate = 1 / (time.time() - startTime)
            print('[{}/{}] time: {}, frame rate: {:0.3f} Hz'.format(i+1, nImgs, \
                    prettyTime(time.time() - startTime), frameRate))
            cv2.imshow('Image', img)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:    break
                                         
        cv2.destroyAllWindows()
                                   
#-------------------------------------------------------------------------------
        
        sess.close()
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.
            
#===============================================================================

    def calcRBCperformance(self, inferDir):
        '''
        We want to explicitly calculate the precision and recall beteween the 
        infected and normal rbc classes, because that is what shows the
        reliability of the system on the malaria detection. 
        This function evaluates the output of the model on a set of images
        located inside the inferDir. The inference on these images are done in a 
        loop. And calculates the precision and recall between the infected and 
        normal rbc classes in these images.
        '''
        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS.
        
        # All the images in the batch are already resized to the appropriate shape 
        # when the batch was created, hence no need to resize again.
        x = tfl.placeholder(dtype=tfl.float32, name='xPlaceholder', \
                            shape=[None, inImgH, inImgW, 3])
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tfl.variable_scope(modelName, reuse=tfl.AUTO_REUSE):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.

            # List of model variables.
            listOfModelVars = []
            for v in tfl.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find(modelName) >= 0:
                    listOfModelVars.append(v)
                    #print('Model: {}, Variable: {}'.format(modelName, v))

#-------------------------------------------------------------------------------

            # Modifying the predictions into proper formats.

            # The raw output of the model has to be converted into proper form 
            # by incorporating the sigmoid and softmax functions with them.
            predLogitsOneHotVec = tfl.nn.softmax(predLogits[:, :, :, :, : nClasses])
            predLogitsXoffset = tfl.sigmoid(predLogits[:, :, :, :, -5])
            predLogitsYoffset = tfl.sigmoid(predLogits[:, :, :, :, -4])
            predLogitsWoffset = tfl.exp(predLogits[:, :, :, :, -3])
            predLogitsHoffset = tfl.exp(predLogits[:, :, :, :, -2])
           
            # The anchor offsets needs to be multiplied with the anchor dimensions.
            # Creating some tensors of those anchor dimensions for this purpose.
            anchorWlist = [anc[0] for anc in anchorList]
            anchorWarr = np.tile(anchorWlist, (finalLayerH, finalLayerW, 1))            
            anchorWtensor = tfl.convert_to_tensor(anchorWarr, dtype=tfl.float32)
            anchorHlist = [anc[1] for anc in anchorList]
            anchorHarr = np.tile(anchorHlist, (finalLayerH, finalLayerW, 1))
            anchorHtensor = tfl.convert_to_tensor(anchorHarr, dtype=tfl.float32)

            predLogitsWoffset = tfl.multiply(predLogitsWoffset, anchorWtensor)
            predLogitsHoffset = tfl.multiply(predLogitsHoffset, anchorHtensor)
            predLogitsConfidence = tfl.sigmoid(predLogits[:, :, :, :, -1])

            # Now combining all these individual predlogits into one tensor.
            predLogitsXoffset = tfl.expand_dims(predLogitsXoffset, axis=-1)
            predLogitsYoffset = tfl.expand_dims(predLogitsYoffset, axis=-1)
            predLogitsWoffset = tfl.expand_dims(predLogitsWoffset, axis=-1)
            predLogitsHoffset = tfl.expand_dims(predLogitsHoffset, axis=-1)
            predLogitsConfidence = tfl.expand_dims(predLogitsConfidence, axis=-1)
            
            predResult = tfl.concat([predLogitsOneHotVec, predLogitsXoffset, \
                                      predLogitsYoffset, predLogitsWoffset, \
                                      predLogitsHoffset, predLogitsConfidence], axis=-1)

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tfl.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt(ckptDirPathDetector, \
                                                              training=self.isTraining)

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tfl.train.Saver(listOfModelVars)
            saver.restore(sess, ckptPath)            
#            print('\nReloaded ALL variables from checkpoint: {}'.format(\
#                                                        ckptPath))
        else:
            # When there are no valid checkpoints.
            print('\nNo valid checkpoints found. Aborting.\n')
            return

        with open(jsonFilePath, 'r') as infoFile:
            infoDict = json.load(infoFile)
                
        # Reloading mean and std from checkpoint.
        mean = np.array(infoDict['mean'])
        std = np.array(infoDict['std'])

#-------------------------------------------------------------------------------

        # Running the loop now.
        listOfImgs = os.listdir(os.path.join(inferDir, 'images'))
        nImgs = len(listOfImgs)
        
        # We also need to calculate the precision and recall between the infected 
        # and normal rbc cells.
        rbcTP, rbcFP, rbcTN, rbcFN = 0.0, 0.0, 0.0, 0.0
        nTotalNormalRbc, nTotalInfectedRbc = 0.0, 0.0
    
        # We also need to calculate the precision and recall between the images
        # containing the infected and normal rbc's.
        rbcImgTP, rbcImgFP, rbcImgTN, rbcImgFN = 0.0, 0.0, 0.0, 0.0
        nNormalRbcImg, nInfectedRbcImg = 0.0, 0.0
    
        startTime = time.time()

        for i in range(nImgs):
            labelDictList, multiHotLabel = getImgLabel(inferDir, listOfImgs[i])
            
            img = cv2.imread(os.path.join(inferDir, 'images', listOfImgs[i]))
            imgBatch = np.array([img])

#           # Normalizing by mean and std as done in case of training.
#           imgBatch = (imgBatch - mean) / std

            # Converting image to range 0 to 1.
            # The image is explicitly converted to float32 to match the type 
            # specified in the placeholder. If img would have been directly divided
            # by 127.5, then it would result in np.float64.
            imgBatch = np.asarray(imgBatch, dtype=np.float32) / 127.5 - 1.0
        
            feedDict = {x: imgBatch}

            inferPredLogits = sess.run(predLogits, feed_dict=feedDict)
            
            inferPredResult = sess.run(predResult, feed_dict=feedDict)
            
            detectedBatchClassScores, _, detectedBatchClassNames, detectedBatchBboxes \
                                                = nonMaxSuppression(inferPredResult)
            
            # The output of the nonMaxSuppression is in the form of a batch.
            # So extracting the contents of this batch since there is an output 
            # of only one image in this batch.        
            detectedBatchClassScores = detectedBatchClassScores[0]
            detectedBatchClassNames = detectedBatchClassNames[0]
            detectedBatchBboxes = detectedBatchBboxes[0]
            
#-------------------------------------------------------------------------------

            # Now we also explicitly calculate the precision and recall beteween
            # the infected and normal rbc classes, because that is what shows the
            # reliability of the system on the malaria detection.
            
            # Also, we need to have a measure of how many images (not individual
            # rbc cells) were truely infected or not. The flag predictedInfection
            # and trueInfection are for that purpose.
            predictedInfection, trueInfection = False, False

            for pdx, p in enumerate(detectedBatchClassNames):
                if p != '_' and p != 'Infected':     continue    # Skip non-rbc classes.
                
                if p == 'Infected':     predictedInfection = True
    
                bx, by, bw, bh = detectedBatchBboxes[pdx].tolist()
                
                # Now trying to find which true bbox will have the max iou with this 
                # predicted bbox, and that corresponding className is termed 
                # bestMatchBboxClassName.
                maxIou, bestMatchBboxClassName = iouThresh, None
                for l in labelDictList:
                    trueName = l['className']
                    if trueName != '_' and trueName != 'Infected':   continue # Skip non-rbc classes.
                    
                    if trueName == 'Infected':  trueInfection = True
                    
                    tlX, tlY, bboxW, bboxH = l['tlX'], l['tlY'], l['bboxW'], l['bboxH']
                    
                    iou, _, _ = findIOU([bx, by, bw, bh], [tlX, tlY, bboxW, bboxH])
                    if iou >= maxIou:   maxIou, bestMatchBboxClassName = iou, trueName
                        
                # Positive means INFECTED.
                # Negative means NORMAL.
                if bestMatchBboxClassName == 'Infected' and p == 'Infected':  rbcTP += 1
                elif bestMatchBboxClassName == 'Infected' and p == '_':  rbcFN += 1
                elif bestMatchBboxClassName == '_' and p == 'Infected':  rbcFP += 1
                elif bestMatchBboxClassName == '_' and p == '_':  rbcTN += 1
                
                # If no proper matches are found then the bestMatchBboxClassName will 
                # remain as None. So just continue in that case.
                # Otherwise see if the bestMatchBboxClassName is same as the predicted 
                # name, which will make it a true positive. 
                # Else a false positive or false negative.
                else:   continue

#-------------------------------------------------------------------------------

            # Counting the number of true rbc's in the image, to get a final 
            # count of how many of them were not detected.            
            for l in labelDictList:
                if l['className'] == '_':   nTotalNormalRbc += 1
                elif l['className'] == 'Infected':  nTotalInfectedRbc += 1
                else:   continue                

#-------------------------------------------------------------------------------

            # Counting the number of true positives and negatives in the images,
            # based on the presence and absence of infection.
            if trueInfection == True and predictedInfection == True:    
                rbcImgTP += 1 ; nInfectedRbcImg += 1
                
            elif trueInfection == True and predictedInfection == False:    
                rbcImgFN += 1 ; nInfectedRbcImg += 1

            elif trueInfection == False and predictedInfection == True:    
                rbcImgFP += 1 ; nNormalRbcImg += 1
                
            elif trueInfection == False and predictedInfection == False:    
                rbcImgTN += 1 ; nNormalRbcImg += 1

            print('\rScanning images. [{}/{}]'.format(i+1, nImgs))

#            cv2.imshow('Image', img)
#            key = cv2.waitKey(1)
#            if key & 0xFF == 27:    break

#-------------------------------------------------------------------------------

        cv2.destroyAllWindows()

        # Calculating the precision, recall (or sensitivity) and specificity of 
        # the rbc cells. This is calculated over the entire dataset.
        rbcPrecision = rbcTP / (rbcTP + rbcFP)
        rbcRecall = rbcTP / (rbcTP + rbcFN)
        rbcSpecificity = rbcTN / (rbcTN + rbcFP)
        rbcF1score = 2 * (rbcPrecision * rbcRecall) / (rbcPrecision + rbcRecall)
        
#-------------------------------------------------------------------------------
        
        # Calculating the precision, recall (or sensitivity) and specificity of 
        # the images based on the presence and absence of the infection. 
        # This is calculated over the entire dataset.
        rbcImgPrecision = rbcImgTP / (rbcImgTP + rbcImgFP)
        rbcImgRecall = rbcImgTP / (rbcImgTP + rbcImgFN)
        rbcImgSpecificity = rbcImgTN / (rbcImgTN + rbcImgFP)
        rbcImgF1score = 2 * (rbcImgPrecision * rbcImgRecall) / (rbcImgPrecision + rbcImgRecall)

#-------------------------------------------------------------------------------

        print('\n\nTime Taken: {}, RBC Precision: {:0.3f}, RBC Recall: {:0.3f}, ' \
               'RBC Specificity: {:0.3f}, RBC F1score: {:0.3f}\n'.format(\
                prettyTime(time.time() - startTime), rbcPrecision, rbcRecall, \
                rbcSpecificity, rbcF1score))

        # Creating the table that shows the percentage of TP, TN, FP, FN and non-detected rbc's.
        nTotalRbc = nTotalNormalRbc + nTotalInfectedRbc
        print('\n\nConfusion Matrix for the RBCs in all the images of {}'.format(inferDir))
        print('-'*100)
        print('|\t{} Set\t|\tPred_Infection\t|\tPred_Normal\t|\tNot_detected\t|'.format(inferDir))
        print('-'*100)
        print('|\tTrue_Infected\t|\t{:0.3f} % (TP)\t|\t{:0.3f} % (FN)\t|\t{:0.3f} %\t|'.format(\
                      (rbcTP / nTotalInfectedRbc * 100), (rbcFN / nTotalInfectedRbc * 100), \
                      ((nTotalInfectedRbc - rbcTP - rbcFN) / nTotalInfectedRbc * 100)))
        print('|\tTrue_Normal\t|\t{:0.3f} % (FP)\t|\t{:0.3f} % (TN)\t|\t{:0.3f} %\t|\n\n'.format(\
                      (rbcFP / nTotalNormalRbc * 100), (rbcTN / nTotalNormalRbc * 100), \
                      ((nTotalNormalRbc - rbcFP - rbcTN) / nTotalNormalRbc * 100)))
        
#-------------------------------------------------------------------------------
        
        print('\n\nRBC Image Precision: {:0.3f}, RBC Image Recall: {:0.3f}, ' \
               'RBC Image Specificity: {:0.3f}, RBC Image F1score: {:0.3f}\n'.format(\
                rbcPrecision, rbcRecall, rbcSpecificity, rbcF1score))

        # Creating the table that shows the percentage of TP, TN, FP, FN on the images
        # based on the infection.
        nRbcImg = nNormalRbcImg + nInfectedRbcImg
        print('\n\nConfusion Matrix for the RBC images of {}'.format(inferDir))
        print('-'*100)
        print('|\t{} Set\t|\tPred_Infection\t|\tPred_Normal\t|\tNot_detected\t|'.format(inferDir))
        print('-'*100)
        print('|\tTrue_Infected\t|\t{:0.3f} % (TP)\t|\t{:0.3f} % (FN)\t|\t{:0.3f} %\t|'.format(\
                      (rbcImgTP / nInfectedRbcImg * 100), (rbcImgFN / nInfectedRbcImg * 100), \
                      ((nInfectedRbcImg - rbcImgTP - rbcImgFN) / nInfectedRbcImg * 100)))
        print('|\tTrue_Normal\t|\t{:0.3f} % (FP)\t|\t{:0.3f} % (TN)\t|\t{:0.3f} %\t|\n\n'.format(\
                      (rbcImgFP / nNormalRbcImg * 100), (rbcImgTN / nNormalRbcImg * 100), \
                      ((nNormalRbcImg - rbcImgFP - rbcImgTN) / nNormalRbcImg * 100)))
        
#-------------------------------------------------------------------------------

        sess.close()
        tfl.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.
            
#===============================================================================


        
        

        
        
        


        
        

        
        
        

        
        

        
        
        


        
        

        
        
        
