[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" /> <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FLoss-Functions-Package-Tensorflow-Keras-PyTorch&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# Loss-Functions-Package-Tensorflow-Keras-PyTorch

This rope implements some popular Loass/Cost/Objective Functions that you can use to train your Deep Learning models.

**With multi-class classification or segmentation, we sometimes use loss functions that calculate the average loss for each class, rather than calculating loss from the prediction tensor as a whole. This kernel is meant as a template reference for the basic code, so all examples calculate loss on the entire tensor, but it should be trivial for you to modify it for multi-class averaging.**

I have provided the implementations in three popular libraries i.e. `tensorflow` `keras` and `pytorch`. Lets get started.

These functions cannot simply be written in NumPy, as they must operate on tensors that also have gradient parameters which need to be calculated throughout the model during backpropagation. According, loss functions must be written using backend functions from the respective model library.

With multi-class classification or segmentation, we sometimes use loss functions that calculate the average loss for each class, rather than calculating loss from the prediction tensor as a whole. This kernel is meant as a template reference for the basic code, so all examples calculate loss on the entire tensor, but it should be trivial for you to modify it for multi-class averaging.
#### For [Learning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras](https://github.com/Mr-TalhaIlyas/Learning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras)
#### For [Evaluation-Metrics-Package-Tensorflow-PyTorch-Keras](https://github.com/Mr-TalhaIlyas/Evaluation-Metrics-Package-Tensorflow-PyTorch-Keras/)
## Necessary Imports
You can import some necessary packages as follows
```python
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Keras
import keras
import keras.backend as K
# Tensorflow
form tensorflow import keras
from tensorflow import keras.backend as K
```
## Weighted Catagorical Cross Entropy Loss

```python
# Tensorflow/Keras
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
   
    weights: numpy array of shape (C,) where C is the number of classes
             np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    """
    weights = K.variable(weights)
        
    def loss(y_true, y_pred, from_logits=False):
        if from_logits:
		y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        	#y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
```
Or you can check the `weighted focal loss` below and set the `gamma=0` and `alpha=1` and it'll work same as `weighted catagorical cross entropy`

## Dice Loss
The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation that can also be modified to act as a loss function:

```python
#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
```
```python
def DiceLoss(y_true, y_pred, smooth=1e-6):
    
    # if you are using this loss for multi-class segmentation then uncomment 
    # following lines
    # if y_pred.shape[-1] <= 1:
    #     # activate logits
    #     y_pred = tf.keras.activations.sigmoid(y_pred)
    # elif y_pred.shape[-1] >= 2:
    #     # activate logits
    #     y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    #     # convert the tensor to one-hot for multi-class segmentation
    #     y_true = K.squeeze(y_true, 3)
    #     y_true = tf.cast(y_true, "int32")
    #     y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    # cast to float32 datatype
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    #flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice
```
## BCE-Dice Loss
This loss combines Dice loss with the standard binary cross-entropy (BCE) loss that is generally the default for segmentation models. Combining the two methods allows for some diversity in the loss, while benefitting from the stability of BCE. The equation for multi-class BCE by itself will be familiar to anyone who has studied logistic regression:
```python
#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
```
```python
class DiceLossMulticlass(nn.Module):
    def __init__(self, weights=None, size_average=False):
        super(mIoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        if self.weights is not None:
            assert self.weights.shape == (targets.shape[1], )

        # make a copy not to change the default weights in the instance of DiceLossMulticlass
        weights = self.weights.copy()

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction images, leave BATCH and NUM_CLASSES
        # (BATCH, NUM_CLASSES, H, W) -> (BATCH, NUM_CLASSES, H * W)
        inputs = inputs.view(inputs.shape[0],inputs.shape[1],-1)
        targets = targets.view(targets.shape[0],targets.shape[1],-1)

        #intersection = (inputs * targets).sum()
        intersection = (inputs * targets).sum(0).sum(1)
        #dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice = (2.*intersection + smooth)/(inputs.sum(0).sum(1) + targets.sum(0).sum(1) + smooth)

        if (weights is None) and self.size_average==True:
            weights = (targets == 1).sum(0).sum(1)
            weights /= weights.sum() # so they sum up to 1

        if weights is not None:
            return 1 - (dice*weights).mean()
        else:
            return 1 - weights.mean()
```

```python
#Tensorflow / Keras
def DiceBCELoss(y_true, y_pred, smooth=1e-6):    
    
    # if you are using this loss for multi-class segmentation then uncomment 
    # following lines
    # if y_pred.shape[-1] <= 1:
    #     # activate logits
    #     y_pred = tf.keras.activations.sigmoid(y_pred)
    # elif y_pred.shape[-1] >= 2:
    #     # activate logits
    #     y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    #     # convert the tensor to one-hot for multi-class segmentation
    #     y_true = K.squeeze(y_true, 3)
    #     y_true = tf.cast(y_true, "int32")
    #     y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    # cast to float32 datatype
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    #flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    
    BCE =  binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE
```
### Weighted BCE and Dice Loss
Combines BCE and Dice loss
```python
# Keras/ Tensorflow
def Weighted_BCEnDice_loss(y_true, y_pred):
    
    # if you are using this loss for multi-class segmentation then uncomment 
    # following lines
    # if y_pred.shape[-1] <= 1:
    #     # activate logits
    #     y_pred = tf.keras.activations.sigmoid(y_pred)
    # elif y_pred.shape[-1] >= 2:
    #     # activate logits
    #     y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    #     # convert the tensor to one-hot for multi-class segmentation
    #     y_true = K.squeeze(y_true, 3)
    #     y_true = tf.cast(y_true, "int32")
    #     y_true = tf.one_hot(y_true, num_class, axis=-1)
       
   
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss
    
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss
```
## HED Loss
I was introduced in holistic edge detector to detect edges/boundaries of objects in https://arxiv.org/pdf/1504.06375.pdf.
```python
# Keras/ Tensorflow
def HED_loss(y_true, y_pred):
    
    #y_true = y_true * 255 # b/c keras generator normalizes images
    if y_pred.shape[-1] <= 1:
        y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
        
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    loss = sigmoid_cross_entropy_balanced(y_pred, y_true) 
    return loss

def sigmoid_cross_entropy_balanced(logits, label, name='cross_entropy_loss'):
    """
    From:

	https://github.com/moabitcoin/holy-edge/blob/master/hed/losses.py

    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    if int(str(tf.__version__)[0]) == 1:
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
    if int(str(tf.__version__)[0]) == 2:
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
```
## Jaccard/Intersection over Union (IoU) Loss
The IoU metric, or Jaccard Index, is similar to the Dice metric and is calculated as the ratio between the overlap of the positive instances between two sets, and their mutual combined values:

Like the Dice metric, it is a common means of evaluating the performance of pixel segmentation models.

```python
#PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
```
```python
#Tensorflow / Keras 
def IoULoss(y_true, y_pred, smooth=1e-6):
    
    # if you are using this loss for multi-class segmentation then uncomment 
    # following lines
    # if y_pred.shape[-1] <= 1:
    #     # activate logits
    #     y_pred = tf.keras.activations.sigmoid(y_pred)
    # elif y_pred.shape[-1] >= 2:
    #     # activate logits
    #     y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    #     # convert the tensor to one-hot for multi-class segmentation
    #     y_true = K.squeeze(y_true, 3)
    #     y_true = tf.cast(y_true, "int32")
    #     y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    # cast to float32 datatype
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    #flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    
    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU
```

## Focal Loss
Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 as a means of combatting extremely imbalanced datasets where positive cases were relatively rare. Their paper "Focal Loss for Dense Object Detection" is retrievable here: https://arxiv.org/abs/1708.02002. In practice, the researchers used an alpha-modified version of the function so I have included it in this implementation.

```python
#PyTorch
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
```
 
```python
#Tensorflow / Keras

def FocalLoss(y_true, y_pred):   
    
    alpha = 0.8
    gamma = 2
    # if you are using this loss for multi-class segmentation then uncomment 
    # following lines
    # if y_pred.shape[-1] <= 1:
    #     # activate logits
    #     y_pred = tf.keras.activations.sigmoid(y_pred)
    # elif y_pred.shape[-1] >= 2:
    #     # activate logits
    #     y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    #     # convert the tensor to one-hot for multi-class segmentation
    #     y_true = K.squeeze(y_true, 3)
    #     y_true = tf.cast(y_true, "int32")
    #     y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    # cast to float32 datatype
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss
```
## Weighted Focal Loss
*in developement*
```python
# TensorFlow/Keras
class WFL():
    '''
    Weighted Focal loss
    '''
    def __init__(self, alpha=0.25, gamma=2, class_weights=None, from_logits=False):
        self.class_weights = class_weights
        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, y_true, y_pred):
        
        if self.from_logits:
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)

        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        
        # cast to float32 datatype
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        
        WCCE = y_true * K.log(y_pred) * self.class_weights
        WFL = (self.alpha * K.pow((1-y_pred), self.gamma)) * WCCE
        # reduce sum -> reduces the loss over number of batches by simply taking sum over all samples
        # reduce mean -> reduces the loss ove number of batches by taking mean of all samples
        # if axis=-1 is given input batch is like B * C then loss will have shape B * 1
        # if axis is None then only 1 scaler value is output
        
        return -tf.math.reduce_sum(WFL, -1) #use this for custom training loop and dviding by global batch size. * (1/GB)
        #return -tf.reduce_mean(WFL, -1) # use this for complie fit keras API
```
## Tversky Loss
This loss was introduced in "Tversky loss function for image segmentationusing 3D fully convolutional deep networks", retrievable here: https://arxiv.org/abs/1706.05721. It was designed to optimise segmentation on imbalanced medical datasets by utilising constants that can adjust how harshly different types of error are penalised in the loss function. From the paper:
**... in the case of α=β=0.5 the Tversky index simplifies to be the same as the Dice coefficient, which is also equal to the F1 score. With α=β=1, Equation 2 produces Tanimoto coefficient, and setting α+β=1 produces the set of Fβ scores. Larger βs weigh recall higher than precision (by placing more emphasis on false negatives).**
To summarise, this loss function is weighted by the constants 'alpha' and 'beta' that penalise false positives and false negatives respectively to a higher degree in the loss function as their value is increased. The beta constant in particular has applications in situations where models can obtain misleadingly positive performance via highly conservative prediction. You may want to experiment with different values to find the optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.

```python
#PyTorch
ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
```

```python
#Tensorflow / Keras
def TverskyLoss(y_true, y_pred, smooth=1e-6):
    
        if y_pred.shape[-1] <= 1:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #5.
            y_pred = tf.keras.activations.sigmoid(y_pred)
            #y_true = y_true[:,:,:,0:1]
        elif y_pred.shape[-1] >= 2:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #3.
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
            y_true = K.squeeze(y_true, 3)
            y_true = tf.cast(y_true, "int32")
            y_true = tf.one_hot(y_true, num_class, axis=-1)
           
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
```

## Focal Tversky Loss

A variant on the Tversky loss that also includes the gamma modifier from Focal Loss.
```python
#PyTorch
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
```

```python
#Tensorflow / Keras
def FocalTverskyLoss(y_true, y_pred, smooth=1e-6):
        

        if y_pred.shape[-1] <= 1:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #5.
            y_pred = tf.keras.activations.sigmoid(y_pred)
            #y_true = y_true[:,:,:,0:1]
        elif y_pred.shape[-1] >= 2:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #3.
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
            y_true = K.squeeze(y_true, 3)
            y_true = tf.cast(y_true, "int32")
            y_true = tf.one_hot(y_true, num_class, axis=-1)
        
        
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = K.pow((1 - Tversky), gamma)
        
        return FocalTversky
```

## Lovasz Hinge Loss

This complex loss function was introduced by Berman, Triki and Blaschko in their paper "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks", retrievable here: https://arxiv.org/abs/1705.08790. It is designed to optimise the Intersection over Union score for semantic segmentation, particularly for multi-class instances. Specifically, it sorts predictions by their error before calculating cumulatively how each error affects the IoU score. This gradient vector is then multiplied with the initial error vector to penalise most strongly the predictions that decreased the IoU score the most. This procedure is detailed by jeandebleu in his excellent summary here.

This code is taken directly from the author's github repo here: https://github.com/bermanmaxim/LovaszSoftmax and all credit is to them.

In this kernel I have implemented the flat variant that uses reshaped rank-1 tensors as inputs for PyTorch. You can modify it accordingly with the dimensions and class number of your data as needed. This code takes raw logits so ensure your model does not contain an activation layer prior to the loss calculation.

I have hidden the researchers' own code below for brevity; simply load it into your kernel for the losses to function. In the case of their tensorflow implementation, I am still working to make it compatible with Keras. There are differences between the Tensorflow and Keras function libraries that complicate this.

```python
#PyTorch
class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, inputs, targets):
        probas = F.softmax(inputs, dim=1) # B*C*H*W -> from logits to probabilities
        return lovasz_softmax(probas, targets, self.classes, self.per_image, self.ignore)

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
```

```python
#Keras
# not working yet
# def LovaszHingeLoss(inputs, targets):
#     return lovasz_hinge_loss(inputs, targets)
```

## Combo Loss
This loss was introduced by Taghanaki et al in their paper "Combo loss: Handling input and output imbalance in multi-organ segmentation", retrievable here: https://arxiv.org/abs/1805.02798. Combo loss is a combination of Dice Loss and a modified Cross-Entropy function that, like Tversky loss, has additional constants which penalise either false positives or false negatives more respectively.

```python
#PyTorch
ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, e, 1.0 - e)       
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo
 ```
 
 ```python
#Tensorflow / Keras
def Combo_loss(y_true, y_pred, smooth=1):
  
  e = K.epsilon()
  if y_pred.shape[-1] <= 1:
    ALPHA = 0.8    # < 0.5 penalises FP more, > 0.5 penalises FN more
    CE_RATIO = 0.5 # weighted contribution of modified CE loss compared to Dice loss
    y_pred = tf.keras.activations.sigmoid(y_pred)
  elif y_pred.shape[-1] >= 2:
    ALPHA = 0.3    # < 0.5 penalises FP more, > 0.5 penalises FN more
    CE_RATIO = 0.7 # weighted contribution of modified CE loss compared to Dice loss
    y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
  
  # cast to float32 datatype
  y_true = K.cast(y_true, 'float32')
  y_pred = K.cast(y_pred, 'float32')
  
  targets = K.flatten(y_true)
  inputs = K.flatten(y_pred)
  
  intersection = K.sum(targets * inputs)
  dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
  inputs = K.clip(inputs, e, 1.0 - e)
  out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
  weighted_ce = K.mean(out, axis=-1)
  combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
  
  return combo
 ```
 
 ### Usage
 Some tips
* Tversky and Focal-Tversky loss benefit from very low learning rates, of the order 5e-5 to 1e-4. They would not see much improvement in my kernels until around 7-10 epochs, upon which performance would improve significantly.

* In general, if a loss function does not appear to be working well (or at all), experiment with modifying the learning rate before moving on to other options.

* You can easily create your own loss functions by combining any of the above with Binary Cross-Entropy or any combination of other losses. Bear in mind that loss is calculated for every batch, so more complex losses will increase runtime.

* Care must be taken when writing loss functions for PyTorch. If you call a function to modify the inputs that doesn't entirely use PyTorch's numerical methods, the tensor will 'detach' from the the graph that maps it back through the neural network for the purposes of backpropagation, making the loss function unusable. Discussion of this is available [here](https://discuss.pytorch.org/t/some-problems-in-custom-loss-functions-and-so-on/36618).


#### Refernces

[RNA Kaggle](https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)
