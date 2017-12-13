from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import pandas as pd
import sys
import imp
import smtplib
import socket
from time import time

def predict_model(iX,model_dir=None,opt_mode='classification'):
    if model_dir==None:
        print("No model to load")
        return
    else:
        save_dir=model_dir
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                saver = tf.train.import_meta_graph(save_dir+".meta")
                saver.restore(s,save_dir)
                dop_dic = {}
                for x in tf.get_default_graph().get_operations():
                    if x.type == 'Placeholder':
                        if "drop_out" in x.name:
                            dop_dic[x.name+":0"]=1.0
                fd={'x:0':iX,'train_test:0':False}
                fd.update(dop_dic)
                if opt_mode=='classification':
                    score,c = s.run(['Softmax:0','ClassPred:0'],feed_dict=fd)
                    return score,c


def test_model(iX,iY,model_dir=None,opt_mode='classification',stats_list=['tp','tn','fp','fn','loss','spe','sen','acc']):
    if model_dir==None:
        print("No model to load")
        return
    else:
        stats_l = []
        for _ in stats_list:
            stats_l.append(_+":0")
        return_dic ={}
        save_dir=model_dir
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                saver = tf.train.import_meta_graph(save_dir+".meta")
                saver.restore(s,save_dir)
                dop_dic = {}
                for x in tf.get_default_graph().get_operations():
                    if x.type == 'Placeholder':
                        if "drop_out" in x.name:
                            dop_dic[x.name+":0"]=1.0
                fd={'x:0':iX,'y:0':iY,'train_test:0':False}
                fd.update(dop_dic)
                if opt_mode=='classification':
                    stats_result = s.run(stats_l,feed_dict=fd)
                    print_s=""
                    for _,sr in enumerate(stats_result):
                        print_s += stats_list[_]+":"+str(sr)+" "
                        return_dic[stats_list[_]]=sr
                    print(print_s)
                elif opt_mode=='regression':
                    lt= s.run('loss:0',feed_dict=fd)
                    print("Loss",lt)
        return return_dic

def print_progres_train(ix,it,bx,bt,lt):
    iter_str = "Iter:"+str(ix)+"/"+str(it)
    batch_str = "Batch:"+str(bx)+"/"+str(bt)
    loss_str = "Loss,Tr:"+str(lt)
    print(iter_str,batch_str,loss_str)



def print_progres_test(ix,it,bx,bt,spex,senx,accx,lt,ls):
    iter_str = "Iter:"+str(ix)+"/"+str(it)
    batch_str = "Batch:"+str(bx)+"/"+str(bt)
    loss_str = "Loss,Tr:"+str(lt)+" Ts:"+str(ls)
    spex=np.round(spex,2)
    accx=np.round(accx,2)
    senx=np.round(senx,2)
    perf_str = 'Spec:'+str(spex)+" Sens:"+str(senx)+" Acc:"+str(accx)
    print(iter_str,batch_str,loss_str,perf_str)


def train_model(ix,iy,cv_args,fc_args,itx=None,ity=None,batch_test=False,stddev_n=0.1,
                iters=10,lr=0.001,lrdf=None,lrdr=10,
                batch_size=32,
                restore=False,save=False,save_name=None,restore_name=None,opt_mode='classification'
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
    dropout_phd = {}
    for _i,_ in enumerate(cv_args):
        _['is_training']=train_bool
        if not 'name_scope' in _.keys():
            _['name_scope']='C'+str(_i)
        if 'drop_out_bool' in _.keys():
            if _['drop_out_bool']==True:
                with tf.name_scope('drop_out'):
                    _['drop_out_ph']=tf.placeholder(tf.float32,name='do_ph_'+_['name_scope'])
                dropout_phd[_['drop_out_ph']]=_['drop_out_v']

    cl.extend(cv_args)
    last_cv = multi_conv(cl)
    fcl = [last_cv]
    for _i,_ in enumerate(fc_args):
        _['is_training']=train_bool
        if not 'name_scope' in _.keys():
            _['name_scope']='FC'+str(_i)
        if 'drop_out_bool' in _.keys():
            if _['drop_out_bool']==True:
                with tf.name_scope('drop_out'):
                    _['drop_out_ph']=tf.placeholder(tf.float32,name='do_ph_'+_['name_scope'])
                dropout_phd[_['drop_out_ph']]=_['drop_out_v']
    fcl.extend(fc_args)
    last_mlp = multi_mlp(fcl)
    
    FCL = last_mlp

    if opt_mode=='classification':
        y_CNN = tf.nn.softmax(FCL,name='Softmax')
        class_pred = tf.argmax(y_CNN,1,name='ClassPred')
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
        acc_,spe_,sen_,tp_,tn_,fp_,fn_ = stats_class(y_CNN,y_)
    elif opt_mode=='regression':
        loss = tf.losses.mean_squared_error(y_,FCL)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    
    save_model=save
    restore_model = restore
    save_dir=save_name
    restore_dir=restore_name
    
    rows = d0
    batches = rows//batch_size
    
    with tf.Session() as s:
        if restore_model==True:
            if restore_name==None:
                print("No model file specified")
                return
            else:
                saver.restore(s,restore_dir)
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
                fd.update(dropout_phd)
                l,t= s.run([loss,train_step,],feed_dict=fd)
                
                if batch_test==True:
                    fd={xi:itx,y_:ity,learning_rate:lr,train_bool:False}
                    fd.update(dropout_phd)
                    if opt_mode=='classification':
                        lt,spe,sen,acc= s.run([loss,spe_,sen_,acc_],feed_dict=fd)
                        print_progres_test(_,iters,_b,batches,spe,sen,acc,l,lt)
                    elif opt_mode=='regression':
                        lt= s.run([loss],feed_dict=fd)
                        print("Loss:",lt)
                else:
                    print_progres_train(_,iters,_b,batches,l)
            bb =_b*batches
            if bb < rows:
                xtb = ix[_b+batch_size:rows,:]
                ytb = iy[_b+batch_size:rows,:]
                fd = {xi:xtb,y_:ytb,learning_rate:lr,train_bool:True}
                fd.update(dropout_phd)
                l,t= s.run([loss,train_step],feed_dict=fd)
                if batch_test==True:
                    fd={xi:itx,y_:ity,learning_rate:lr,train_bool:False}
                    fd.update(dropout_phd)
                    if opt_mode=='classification':
                        lt,spe,sen,acc= s.run([loss,spe_,sen_,acc_],feed_dict=fd)
                        print_progres_test(_,iters,_b,batches,spe,sen,acc,l,lt)
                    elif opt_mode=='regression':
                        lt= s.run([loss],feed_dict=fd)
                        print("Loss:",lt)
                else:
                    print_progres_train(_,iters,_b,batches,l)
        if save_model==True:
            if save_name==None:
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
             max_bool=False,max_kernel=[1,2,2,1],max_strides=[1,1,1,1], max_padding='SAME',
             drop_out_bool=False,drop_out_ph=None,drop_out_v=None
             ):
    with tf.name_scope(name_scope):
        input_depth=input_matrix.get_shape().as_list()[3]
        W = tf.Variable(tf.truncated_normal([filter_size,filter_size,input_depth,layer_depth], stddev=stddev_n),name='W')
        b = tf.Variable(tf.constant(stddev_n, shape=[layer_depth]),name='b')
        c = tf.nn.conv2d(input_matrix, W, strides=strides, padding=padding,name='conv') + b
        n = tf.contrib.layers.batch_norm(c, center=True, scale=True, is_training=is_training)
        a = tf.nn.relu(n,name="activation")
        if max_bool==True:
            out = tf.nn.max_pool(a, ksize=max_kernel,strides=max_strides, padding=max_padding,name='max')
        else:
            out = a
        if drop_out_bool==True:
            out_  = tf.nn.dropout(out, drop_out_ph)
        else:
            out_ = out
        return out_


def fc(input_matrix,n=22,norm=False,prev_conv=False,
       stddev_n = 0.05,is_training=True,
       name_scope='FC',drop_out_bool=False,drop_out_ph=None,drop_out_v=None):
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
            out = tf.nn.relu(n,name="activation")
        else:
            out = tf.nn.relu(fc,name="activation")
        if drop_out_bool==True:
            out_  = tf.nn.dropout(out, drop_out_ph)
        else:
            out_ = out
        return out_


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

def acc_sen_spe(tp,tn,fp,fn):
    stats_dic={}
    stats_dic['sen']=(tp/(tp+fn))
    stats_dic['acc']=(tn+tp)/(tp+tn+fp+fn)
    stats_dic['spe']=tn/(tn+tp)
    return stats_dic.copy()

def get_previous_features(i_layer):
    convx_dims = i_layer.get_shape().as_list()
    output_features = 1
    for dim in range(1,len(convx_dims)):
        output_features=output_features*convx_dims[dim]
    return output_features


def send_mail(email_origin,email_destination,email_pass,subject="Test report",content="Test"):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    #Next, log in to the server
    server.login(email_origin,email_pass)
    msg = "Subject:"+subject+" \n\n "+content+"\n" # The /n separates the message from the headers
    server.sendmail(email_origin,email_destination, msg)

def test_model_by_batch(iX,iY,batch_size=16,model_dir=None,opt_mode='classification',stats_list=['tp','tn','fp','fn']):
    xts = iX.shape[0]
    bs = xts//batch_size
    stats_l = []
    for _ in range(0,bs):
        print(str(_)+"/"+str(bs))
        bb = _*batch_size
        bb1 = (1+_)*batch_size
        bx = iX[bb:bb1,:,:,:]
        by = iY[bb:bb1,:]
        stats_dic = test_model(iX=bx,iY=by,model_dir=model_dir,opt_mode=opt_mode,stats_list=stats_list)
        stats_l.append(stats_dic)
    if bs*batch_size<xts:
        print(str(_+1)+"/"+str(bs))
        bb = (1+_)*batch_size
        bb1 = xts
        bx = iX[bb:bb1,:,:,:]
        by = iY[bb:bb1,:]
        stats_dic = test_model(iX=bx,iY=by,model_dir=model_dir,opt_mode=opt_mode,stats_list=stats_list)
        stats_l.append(stats_dic)
    stats_df = pd.DataFrame(stats_l)
    stats_dic = stats_df.sum().to_dict()
    ac_se_sp = acc_sen_spe(**stats_dic)

    stats_dic.update(ac_se_sp)
    return stats_dic


    