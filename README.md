[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" /> <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" />
# Loss-Functions-Package-Tensorflow-Keras-PyTorch

This rope implements some popular Loass/Cost/Objective Functions that you can use to train your Deep Learning models.

**With multi-class classification or segmentation, we sometimes use loss functions that calculate the average loss for each class, rather than calculating loss from the prediction tensor as a whole. This kernel is meant as a template reference for the basic code, so all examples calculate loss on the entire tensor, but it should be trivial for you to modify it for multi-class averaging.**

I have provided the implementations in three popular libraries i.e. `tensorflow` `keras` and `pytorch`. Lets get started.

These functions cannot simply be written in NumPy, as they must operate on tensors that also have gradient parameters which need to be calculated throughout the model during backpropagation. According, loss functions must be written using backend functions from the respective model library.

With multi-class classification or segmentation, we sometimes use loss functions that calculate the average loss for each class, rather than calculating loss from the prediction tensor as a whole. This kernel is meant as a template reference for the basic code, so all examples calculate loss on the entire tensor, but it should be trivial for you to modify it for multi-class averaging.
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
#Tensorflow / Keras
def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
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
def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
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

def Weighted_BCEnDice_loss(y_true, y_pred):
    
    if y_pred.shape[-1] <= 1:
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:
       y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
       y_true = K.squeeze(y_true, 3)
       y_true = tf.cast(y_true, "int32")
       y_true = tf.one_hot(y_true, num_class, axis=-1)
       
   
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
```
## HED Loss
I was introduced in holistic edge detector to detct edges/boundaries of objects in https://arxiv.org/pdf/1504.06375.pdf.
```python
# Keras/ Tensorflow
def HED_loss(y_true, y_pred):
    
    #y_true = y_true * 255#(num_class + 1)  
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
#Keras
def IoULoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
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
#Keras
ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss
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
#Keras
ALPHA = 0.5
BETA = 0.5

def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
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
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
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
class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)    
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
        return Lovasz
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
 #Keras
ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

def Combo_loss(targets, inputs):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
    
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

[RNA Kaggle](https://discuss.pytorch.org/t/some-problems-in-custom-loss-functions-and-so-on/36618)
