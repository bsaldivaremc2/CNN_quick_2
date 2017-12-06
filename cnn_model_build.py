from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import pandas as pd
import sys
import imp

def predict_model(iX,model_dir=None):
    if model_dir==None:
        print("No model to load")
        return
    else:
        save_dir=model_dir
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                saver = tf.train.import_meta_graph(save_dir+".meta")
                saver.restore(s,save_dir)
                fd={'x:0':iX,'train_test:0':False}
                score,c = s.run(['Softmax:0','ClassPred:0'],feed_dict=fd)
                print("Score",score)
                print("Class:",c)
    return score,c
def test_model(iX,iY,model_dir=None):
    if model_dir==None:
        print("No model to load")
        return
    else:
        save_dir=model_dir
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                saver = tf.train.import_meta_graph(save_dir+".meta")
                saver.restore(s,save_dir)
                fd={'x:0':iX,'y:0':iY,'train_test:0':False}
                lt,spe,sen,acc= s.run(['cross_entropy:0','spe:0','sen:0','acc:0'],feed_dict=fd)
                spe=np.round(spe,2)
                acc=np.round(acc,2)
                sen=np.round(sen,2)
                print("Cross entropy",lt,"Specificity",spe,"Sensitivity",sen,"Accuracy",acc)

def print_progres_train(ix,it,bx,bt,lt):
    iter_str = "Iter:"+str(ix)+"/"+str(it)
    batch_str = "Batch:"+str(bx)+"/"+str(bt)
    loss_str = "Loss,Tr:"+str(lt)
    print(iter_str,batch_str,loss_str)

def print_progres_test(ix,it,bx,bt,spex,senx,accx,lt,ls):
    iter_str = "Iter:"+str(ix)+"/"+str(it)
    batch_str = "Batch:"+str(bx)+"/"+str(bt)
    loss_str = "Loss,Tr:"+str(lt)+" Ts:"+str(ls)
    perf_str = 'Spec:'+str(spex)+" Sens:"+str(senx)+" Acc:"+str(accx)
    print(iter_str,batch_str,loss_str,perf_str)


def train_model(ix,iy,cv_args,fc_args,itx=None,ity=None,batch_test=False,stddev_n=0.1,
                iters=10,lr=0.001,lrdf=None,lrdr=10,
                batch_size=32,
                restore=False,save=False,model_name=None
               ):
    
    tf.reset_default_graph()
    class_output = iy.shape[1]
    d0 = ix.shape[0]
    d1 = ix.shape[1]
    d2 = ix.shape[2]
    d3 = ix.shape[3]
    xi = tf.placeholder(tf.float32, shape=[None,d1,d2,d3],name='x')
    #x_ = tf.reshape(xi,[-1,28,28,1])
    y_ = tf.placeholder(tf.float32, shape=[None,class_output],name='y')
    train_bool=tf.placeholder(bool,name='train_test')
    learning_rate = tf.placeholder(tf.float32)
    
    cl= [xi]
    for _ in cv_args:
        _['is_training']=train_bool
    cl.extend(cv_args)
    last_cv = multi_conv(cl)
    fcl = [last_cv]
    for _ in fc_args:
        _['is_training']=train_bool
    fcl.extend(fc_args)
    last_mlp = multi_mlp(fcl)
    
    FCL_input = last_mlp
    FCL_input_features = get_previous_features(FCL_input)
    W_FCL = tf.Variable(tf.truncated_normal([FCL_input_features, class_output], stddev=stddev_n))
    b_FCL = tf.Variable(tf.constant(stddev_n, shape=[class_output])) 
    FCL=tf.matmul(FCL_input, W_FCL) + b_FCL
    y_CNN = tf.nn.softmax(FCL,name='Softmax')
    class_pred = tf.argmax(y_CNN,1,name='ClassPred')
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="cross_entropy")
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    acc_,spe_,sen_,tp_,tn_,fp_,fn_ = stats_class(y_CNN,y_)
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    
    save_model=save
    restore_model = restore
    save_dir=model_name
    
    rows = d0
    batches = rows//batch_size
    
    with tf.Session() as s:
        if restore_model==True:
            if model_name==None:
                print("No model file specified")
                return
            else:
                saver.restore(s,save_dir)
        else:
            s.run(init_op)
    
        print("Start training")
        for _ in range(0,iters):
            if lrdf!=None:
                if (_ > 0) and (_%lrdf==0):
                    lr=lr/lrdr        
            for _b in range(0,batches):
                xtb = ix[_b:_b+batch_size,:]
                ytb = iy[_b:_b+batch_size,:]
                fd = {xi:xtb,y_:ytb,learning_rate:lr,train_bool:True}
                l,t= s.run([cross_entropy,train_step,],feed_dict=fd)
                
                if batch_test==True:
                    fd={xi:itx,y_:ity,learning_rate:lr,train_bool:False}
                    lt,spe,sen,acc= s.run([cross_entropy,spe_,sen_,acc_],feed_dict=fd)
                    spe=np.round(spe,2)
                    acc=np.round(acc,2)
                    sen=np.round(sen,2)
                    print_progres_test(_,iters,_b,batches,spe,sen,acc,l,lt)
                else:
                    print_progres_train(_,iters,_b,batches,l)
            bb =_b*batches
            if bb < rows:
                xtb = ix[_b+batch_size:rows,:]
                ytb = iy[_b+batch_size:rows,:]
                fd = {xi:xtb,y_:ytb,learning_rate:lr,train_bool:True}
                l,t= s.run([cross_entropy,train_step],feed_dict=fd)
                if batch_test==True:
                    fd={xi:itx,y_:ity,learning_rate:lr,train_bool:False}
                    lt,spe,sen,acc= s.run([cross_entropy,spe_,sen_,acc_],feed_dict=fd)
                    spe=np.round(spe,2)
                    acc=np.round(acc,2)
                    sen=np.round(sen,2)
                    print_progres_test(_,iters,_b,batches,spe,sen,acc,l,lt)
                else:
                    print_progres_train(_,iters,_b,batches,l)
        if save_model==True:
            if model_name==None:
                print("No model specified, model not being saved")
                return
            else:
                save_path = saver.save(s, save_dir)
                print("Model saved in file: %s" % save_path)
   
def multi_mlp(i_params):
    il = i_params
    mlp_dic = {'FC0':il[0]}
    for _ in range(1,len(il)):
        params = il[_]
        mlp_dic['FC'+str(_)]=fc(mlp_dic['FC'+str(_-1)],**params)
    return mlp_dic['FC'+str(_)]

def multi_conv(i_conv_params):
    il = i_conv_params
    conv_dic = {'C0':il[0]}
    for _ in range(1,len(il)):
        params = il[_]
        conv_dic['C'+str(_)]=conv(conv_dic['C'+str(_-1)],**params)
    return conv_dic['C'+str(_)]

def conv(input_matrix,filter_size=3,layer_depth=8,
              strides=[1,1,1,1],padding='SAME',
              is_training=True,name_scope="lx",
              stddev_n = 0.05,
             max_bool=False,max_kernel=[1,2,2,1],max_strides=[1,1,1,1], max_padding='SAME'
             ):
    with tf.name_scope(name_scope):
        input_depth=input_matrix.get_shape().as_list()[3]
        W = tf.Variable(tf.truncated_normal([filter_size,filter_size,input_depth,layer_depth], stddev=stddev_n),name='W')
        b = tf.Variable(tf.constant(stddev_n, shape=[layer_depth]),name='b')
        c = tf.nn.conv2d(input_matrix, W, strides=strides, padding=padding,name='conv') + b
        n = tf.contrib.layers.batch_norm(c, center=True, scale=True, is_training=is_training)
        a = tf.nn.relu(n,name="activation")
        if max_bool==True:
            return tf.nn.max_pool(a, ksize=max_kernel,strides=max_strides, padding=max_padding,name='max')
        else:
            return a


def fc(input_matrix,n=22,norm=False,prev_conv=False,
       stddev_n = 0.05,is_training=True,
       name_scope='FC'):
    with tf.name_scope(name_scope):
        cvpfx = get_previous_features(input_matrix)
        if prev_conv==True:
            im = tf.reshape(input_matrix, [-1, cvpfx])
        else:
            im = input_matrix
        W = tf.Variable(tf.truncated_normal([cvpfx, n], stddev=stddev_n),name='W')
        b = tf.Variable(tf.constant(stddev_n, shape=[n]),name='b') 
        fc = tf.matmul(im, W) + b
        if norm==True:
            n = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=is_training)
            return tf.nn.relu(n,name="activation")
        else:
            return tf.nn.relu(fc,name="activation")


def stats_class(predicted,ground_truth):
    yi = tf.argmax(ground_truth,1)
    yp = tf.argmax(predicted,1)
    tpi = yp*yi
    tp = tf.reduce_sum(tf.cast(tf.greater(tpi,0),tf.int32),name='tp')
    fni = yi-yp
    fn = tf.reduce_sum(tf.cast(tf.greater(fni,0),tf.int32),name='fn')
    sensitivity = tf.divide(tp,(fn+tp),name='sen')    #sensitivity = tp/(fn+tp)    
    tni = yi+yp
    tn = tf.reduce_sum(tf.cast(tf.equal(tni,0),tf.int32),name='tn')    
    fpi = yp - yi
    fp = tf.reduce_sum(tf.cast(tf.greater(fpi,0),tf.int32),name='fp')
    specificity = tf.divide(tn,(tn+fp),name='spe')#specificity = tn/(tn+fp)
    accuracy = tf.divide((tn+tp),(tn+tp+fn+fp),name='acc')#accuracy = (tn+tp)/(tn+tp+fn+fp)
    return [accuracy,specificity,sensitivity,tp,tn,fp,fn]


def get_previous_features(i_layer):
    convx_dims = i_layer.get_shape().as_list()
    output_features = 1
    for dim in range(1,len(convx_dims)):
        output_features=output_features*convx_dims[dim]
    return output_features



