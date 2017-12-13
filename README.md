
# Test fast CNN model building

With this small library it is possible to deploy CNN quickly. Similarly to the definition of a Keras object, nonetheless, this library will be oriented to export it for a production like environment. For now, classification with Softmax.

### Example of usage

Looading the library and other libraries. Later, transforming the data to have the usable format of the library. X_train to [m_rows,pic_height,pic_witdh,channels]


```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import pandas as pd
import sys
import imp

import cnn_model_build
imp.reload(cnn_model_build)

train_file = "train.csv"
train_df = pd.read_csv(train_file)

X = train_df.iloc[:,1:].values
Y = pd.get_dummies(train_df['label']).values

print(X.shape,Y.shape)

train_dev_prop = 0.8
train_dev_ind = int(X.shape[0]*train_dev_prop)
X_train = X[0:train_dev_ind,:]
Y_train = Y[0:train_dev_ind,:]

X_dev = X[train_dev_ind:X.shape[0],:]
Y_dev = Y[train_dev_ind:X.shape[0],:]

print(X_train.shape,Y_train.shape,X_dev.shape,Y_dev.shape)

X_test = X_dev
Y_test = Y_dev

X_train = X_train.reshape(X_train.shape[0],int(X_train.shape[1]**0.5),int(X_train.shape[1]**0.5),1)
X_test = X_test.reshape(X_test.shape[0],int(X_test.shape[1]**0.5),int(X_test.shape[1]**0.5),1)

```

    (42000, 784) (42000, 10)
    (33600, 784) (33600, 10) (8400, 784) (8400, 10)


## Training model

Here we define two lists, one for the convolutional layers and the other for the fully connected layers. For every item we put in them, we specify the params to setup each one. The **name_scope** param is very important since it allows us to re-use variables in the library. Nevertheless, it can be ommited. Conv layers have batch-norm activated by default.   
When it is training it learns the exponential weighted average using the tensorflow contrib layers. For testing and predictions uses the **is_training** variable as **False**.   
**lr** is learning rate. **lrdr** is to indicate after how many iters the learning rate will decrease by a factor of **lrdf**.   
**Save** and **restore** will perform those actions using the **model_name** file. If **batch_test** is set to True, **itx** and **ity** arrays will be used for testing and report the **accuracy**, **cross entropy**, **specificity** and **sensitivity**, for every batch processing.  
The last layer to be added should be a fully connected with the size equal to the amount of classes for a classification or points for a regression.


```python
cvl=[]
cvl.append({'filter_size':2,'name_scope':'C1'})
cvl.append({'filter_size':2,'name_scope':'C2','max_bool':True,'max_strides':[1,2,2,1]})
cvl.append({'filter_size':2,'name_scope':'C3','max_bool':True,'max_strides':[1,2,2,1]})
fcl = []
fcl.append({'n':5,'prev_conv':True,'norm':True,'name_scope':'FC1'})
fcl.append({'n':10,'norm':True})
fcl.append({'n':10,'norm':True})
fcl.append({'n':10,'name_scope':'FCL'})
cnn_model_build.train_model(ix=X_train,iy=Y_train,cvargs=cvl,fc_args=fcl,
            itx=X_test,ity=Y_test,batch_test=False,
            stddev_n=0.1,
                iters=1,lr=0.001,lrdf=None,lrdr=None,
                batch_size=256,
                restore=False,save=True,model_name='MNIST_NL/mnis_01.ckpt'
               )


```

## Testing model

To test the model and use a testing array independently of the training process, cnn_model_build.test_model should be used. This function will load the saved model specified in **model_dir** to get the accuracy, cross entropy, specificity and sensitivity.


```python
stats = cnn_model_build.test_model(iX=X_test,iY=Y_test,model_dir='MNIST_NL/mnis_01.ckpt')
```

## Predicting with the model

In order to make predictions with the trained model, even with a single example, you can use **cnn_model_build.predict_model**. This function returns the score, which is the result of the softmax layer and the class prediction, which is the column number  starting by zero, of the classes trained.


```python
score, class_pred = cnn_model_build.predict_model(iX=X_test[0:1,:,:,:],model_dir='MNIST_NL/mnis_01.ckpt')
```
## Train the model and test using mini-batch. Send emails to report progress and end of process  

You can change the depth, number of filters, of the CNN layers using **layer_depth**. In addition you can enable dropout by setting **drop_out_bool** to **True** and setting the value for it **drop_out_v**. It can be added in Convolutional layers and in Fully connected.  

You can save and restores models using different locations. Restore and save to another directory or file. Use **save_name** and **restore_name** for this purpose.  
  
  
The function **test_model_by_batch** allows to split into batches the dev/testing set when a full batch test isn't possible. This function will return a dictionary with the **True positives (tp), True negatives (tn), False positives (fp), False negatives (fn), accuracy (acc), sensitivity (sen) and specificity (spe)**.   

The function **send_email** included could be used to report the training/test progress and when the it is done, as can be seen lines below.  


```python


cvl=[]
cvl.append({'filter_size':3,'layer_depth':32})
cvl.append({'filter_size':3,'layer_depth':32,
           'max_bool':True,'max_strides':[1,2,2,1]})
cvl.append({'filter_size':3,'layer_depth':512,
           'max_bool':True,'max_strides':[1,2,2,1],
           'drop_out_bool':True,'drop_out_v':0.9})
fcl = []
fcl.append({'n':128,'prev_conv':True,'norm':True,
            'drop_out_bool':True,'drop_out_v':0.5})
fcl.append({'n':10,'name_scope':'FCL'})


batch_size = 16
iters=8
lr=0.00001
restore=True
save=True
repeat_times = 20
model_name='ModelX/modelx_v01.ckpt'
eo='sender.email@gmail.com'
ed='destination.email@gmail.com'
pwd='YourPasswordForSenderEmail'
sjt="Subject of the email "

for _ in range(0,repeat_times):
    if _==0:
        restore=False
    else:
        restore=True
    cnn_model_build.train_model(x_train,y_train,cv_args=cvl,fc_args=fcl,batch_test=False,
            stddev_n=0.1,iters=iters,lr=lr,lrdf=None,lrdr=None,batch_size=batch_size,
    restore=restore,save=save,save_name=model_name,restore_name=model_name)
    stats = cnn_model_build.test_model_by_batch(x_test,y_test,batch_size=batch_size,model_dir=model_name)
    _i = (_+1)*iters
    _t = repeat_times*iters
    ctnt=" Iter:"+str(_i)+"/"+str(_t)+" lr:"+str(lr)+"\n"
    ctnt+="Test set:"+str(stats)+" \n"+" model_name"+model_name
    cnn_model_build.send_mail(eo,ed,pwd,sjt,ctnt)


cnn_model_build.send_mail(eo,ed,pwd,sjt,"Finished")




```