# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx

bn_mom = 0.9

def get_symbol(stage_num,lambda_local,lambda_d,**kwargs):
    global bn_mom
    bn_mom = kwargs.get('bn_mom', 0.9)
    wd_mult = kwargs.get('wd_mult', 1.)
    # stage_num=[1,1,1]
    # lambda_local = 0
    # lambda_d = 0


    data = mx.symbol.Variable(name="data") # 224
    data = data-127.5
    data = data*0.0078125
    # version_input = kwargs.get('version_input', 1)
    # assert version_input>=0
    # version_output = kwargs.get('version_output', 'E')
    # fc_type = version_output
    # version_unit = kwargs.get('version_unit', 3)
    # print(version_input, version_output, version_unit)
    #-------------------------------------------------------------------------------------------------------------
    x=mx.sym.Convolution(data=data,num_filter=32,kernel=(3,3))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    x=mx.sym.Activation(x,act_type='relu')
    x_layer1 = mx.sym.Pooling(data=x, kernel=(2, 2),stride=(2,2), pool_type='avg', name='pool1')
    x=mx.sym.Convolution(data=x_layer1,num_filter=32,kernel=(3,3))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    x=mx.sym.Activation(data=x,act_type='relu')
    x_layer2=mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2,2),pool_type='avg', name='pool2')
    x=mx.sym.Convolution(data=x_layer2,num_filter=32,kernel=(3,3))
    x=mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    x=mx.sym.Activation(data=x,act_type='relu')
    x_layer3=mx.sym.Pooling(data=x, kernel=(2, 2),stride=(2,2), pool_type='avg', name='pool3')
    x=mx.sym.Convolution(data=x_layer3,num_filter=32,kernel=(3,3))
    x=mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    x=mx.sym.Activation(data=x,act_type='relu')
    #-------------------------------------------------------------------------------------------------------------
    s=mx.sym.Convolution(data=data,num_filter=16,kernel=(3,3))
    s=mx.sym.BatchNorm(data=s, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    s=mx.sym.Activation(data=s,act_type='tanh')
    s_layer1=mx.sym.Pooling(data=s, kernel=(2, 2), stride=(2,2),pool_type='max', name='pool4')
    s=mx.sym.Convolution(data=s_layer1,num_filter=16,kernel=(3,3))
    s=mx.sym.BatchNorm(data=s, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    s=mx.sym.Activation(data=s,act_type='tanh')
    s_layer2=mx.sym.Pooling(data=s, kernel=(2, 2), stride=(2,2),pool_type='max', name='pool5')
    s=mx.sym.Convolution(data=s_layer2,num_filter=16,kernel=(3,3))
    s=mx.sym.BatchNorm(data=s, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    s=mx.sym.Activation(data=s,act_type='tanh')
    s_layer3=mx.sym.Pooling(data=s, kernel=(2, 2),stride=(2,2), pool_type='max', name='pool6')
    s=mx.sym.Convolution(data=s_layer3,num_filter=16,kernel=(3,3))
    s=mx.sym.BatchNorm(data=s, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    s=mx.sym.Activation(data=s,act_type='tanh')
    #-------------------------------------------------------------------------------------------------------------
    # Classifier block
    s_layer4=mx.sym.Convolution(data=s,num_filter=10,kernel=(1,1))
    s_layer4=mx.sym.Flatten(data=s_layer4)
    s_layer4_mix=mx.sym.Dropout(data=s_layer4,p=0.2)
    s_layer4_mix=mx.sym.FullyConnected(data=s_layer4_mix,num_hidden=stage_num[0])
    s_layer4_mix=mx.sym.Activation(data=s_layer4_mix,act_type='relu')

    x_layer4=mx.sym.Convolution(data=x,num_filter=10,kernel=(1,1))
    x_layer4=mx.sym.Flatten(data=x_layer4)
    x_layer4_mix=mx.sym.Dropout(data=x_layer4,p=0.2)
    x_layer4_mix=mx.sym.FullyConnected(data=x_layer4_mix,num_hidden=stage_num[0])
    x_layer4_mix=mx.sym.Activation(data=x_layer4_mix,act_type='relu')

    feat_a_s1_pre=s_layer4*x_layer4
    delta_s1=mx.sym.FullyConnected(data=feat_a_s1_pre,num_hidden=1,name='delta_s1')
    delta_s1=mx.sym.Activation(data=delta_s1,act_type='tanh')

    feat_a_s1=s_layer4_mix*x_layer4_mix
    feat_a_s1=mx.sym.FullyConnected(data=feat_a_s1,num_hidden=2*stage_num[0])
    feat_a_s1=mx.sym.Activation(data=feat_a_s1,act_type='relu')
    pred_a_s1=mx.sym.FullyConnected(data=feat_a_s1,num_hidden=stage_num[0],name='pred_age_stage1')
    pred_a_s1=mx.sym.Activation(data=pred_a_s1,act_type='relu')

    local_s1=mx.sym.FullyConnected(data=feat_a_s1,num_hidden=stage_num[0],name='local_delta_stage1')
    local_s1=mx.sym.Activation(data=local_s1,act_type='tanh')
    #-------------------------------------------------------------------------------------------------------------
    s_layer2=mx.sym.Convolution(data=s_layer2,num_filter=10,kernel=(1,1))
    s_layer2=mx.sym.Activation(data=s_layer2,act_type='relu')
    s_layer2=mx.sym.Pooling(data=s_layer2, kernel=(4, 4), stride=(4,4),pool_type='max', name='pool7')
    s_layer2=mx.sym.Flatten(data=s_layer2)
    s_layer2_mix=mx.sym.Dropout(data=s_layer2,p=0.2)
    s_layer2_mix=mx.sym.FullyConnected(data=s_layer2_mix,num_hidden=stage_num[1])
    s_layer2_mix=mx.sym.Activation(data=s_layer2_mix,act_type='relu')

    x_layer2=mx.sym.Convolution(data=x_layer2,num_filter=10,kernel=(1,1))
    x_layer2=mx.sym.Activation(data=x_layer2,act_type='relu')
    x_layer2=mx.sym.Pooling(data=x_layer2, kernel=(4, 4), stride=(4,4),pool_type='avg', name='pool8')
    x_layer2=mx.sym.Flatten(data=x_layer2)
    x_layer2_mix=mx.sym.Dropout(data=x_layer2,p=0.2)
    x_layer2_mix=mx.sym.FullyConnected(data=x_layer2_mix,num_hidden=stage_num[1])
    x_layer2_mix=mx.sym.Activation(data=x_layer2_mix,act_type='relu')

    feat_a_s2_pre=s_layer2*x_layer2
    delta_s2=mx.sym.FullyConnected(data=feat_a_s2_pre,num_hidden=1,name='delta_s2')
    delta_s2=mx.sym.Activation(data=delta_s2,act_type='tanh')

    feat_a_s2=s_layer2_mix*x_layer2_mix
    feat_a_s2=mx.sym.FullyConnected(data=feat_a_s2,num_hidden=2*stage_num[1])
    feat_a_s2=mx.sym.Activation(data=feat_a_s2,act_type='relu')
    pred_a_s2=mx.sym.FullyConnected(data=feat_a_s2,num_hidden=stage_num[1],name='pred_age_stag2')
    pred_a_s2=mx.sym.Activation(data=pred_a_s2,act_type='relu')
    local_s2=mx.sym.FullyConnected(data=feat_a_s2,num_hidden=stage_num[1],name='local_delta_stage2')
    local_s2=mx.sym.Activation(data=local_s2,act_type='tanh')
    #-------------------------------------------------------------------------------------------------------------
    s_layer1=mx.sym.Convolution(data=s_layer1,num_filter=10,kernel=(1,1))
    s_layer1=mx.sym.Activation(data=s_layer1,act_type='relu')
    s_layer1=mx.sym.Pooling(data=s_layer1, kernel=(8, 8),stride=(8,8), pool_type='max', name='pool9')
    s_layer1=mx.sym.Flatten(data=s_layer1)
    s_layer1_mix=mx.sym.Dropout(data=s_layer1,p=0.2)
    s_layer1_mix=mx.sym.FullyConnected(data=s_layer1_mix,num_hidden=stage_num[2])
    s_layer1_mix=mx.sym.Activation(data=s_layer1_mix,act_type='relu')

    x_layer1=mx.sym.Convolution(data=x_layer1,num_filter=10,kernel=(1,1))
    x_layer1=mx.sym.Activation(data=x_layer1,act_type='relu')
    x_layer1=mx.sym.Pooling(data=x_layer1, kernel=(8, 8), stride=(8,8),pool_type='avg', name='pool10')
    x_layer1=mx.sym.Flatten(data=x_layer1)
    x_layer1_mix=mx.sym.Dropout(data=x_layer1,p=0.2)
    x_layer1_mix=mx.sym.FullyConnected(data=x_layer1_mix,num_hidden=stage_num[2])
    x_layer1_mix=mx.sym.Activation(data=x_layer1_mix,act_type='relu')

    feat_a_s3_pre=s_layer1*x_layer1
    delta_s3=mx.sym.FullyConnected(data=feat_a_s3_pre,num_hidden=1,name='delta_s3')
    delta_s3=mx.sym.Activation(data=delta_s3,act_type='tanh')

    feat_s_s3=s_layer1_mix*x_layer1_mix
    feat_a_s3=mx.sym.FullyConnected(data=feat_s_s3,num_hidden=2*stage_num[2])
    feat_a_s3=mx.sym.Activation(data=feat_a_s3,act_type='relu')
    pred_a_s3=mx.sym.FullyConnected(data=feat_a_s3,num_hidden=stage_num[2],name='pred_age_stage3')
    pred_a_s3=mx.sym.Activation(data=pred_a_s3,act_type='relu')

    local_s3=mx.sym.FullyConnected(data=feat_a_s3,num_hidden=stage_num[2],name='local_delta_stage3')
    local_s3=mx.sym.Activation(data=local_s3,act_type='tanh')
    #-------------------------------------------------------------------------------------------------------------
    i1=mx.symbol.arange(0,stage_num[0])
    i1=mx.symbol.expand_dims(i1,axis=0)
    a=mx.symbol.broadcast_add(i1,lambda_local*local_s1)*pred_a_s1
    a=mx.symbol.sum(a,axis=1,keepdims=True)
    # a=mx.symbol.expand_dims(a,axis=1)
    a=a/(stage_num[0]*(1+lambda_d*delta_s1))

    i2=mx.symbol.arange(0,stage_num[1])
    i2=mx.symbol.expand_dims(i2,axis=0)
    b=mx.symbol.broadcast_add(i2,lambda_local*local_s2)*pred_a_s2
    b=mx.symbol.sum(b,axis=1,keepdims=True)
    # b=mx.symbol.expand_dims(b,axis=1)
    b=b/(stage_num[0]*(1+lambda_d*delta_s1))/(stage_num[1]*(1+lambda_d*delta_s2))

    i3=mx.symbol.arange(0,stage_num[2])
    i3=mx.symbol.expand_dims(i3,axis=0)
    c=mx.symbol.broadcast_add(i3,lambda_local*local_s3)*pred_a_s3
    c=mx.symbol.sum(c,axis=1,keepdims=True)
    # c=mx.symbol.expand_dims(c,axis=1)
    c=c/(stage_num[0]*(1+lambda_d*delta_s1))/(stage_num[1]*(1+lambda_d*delta_s2))/(stage_num[2]*(1+lambda_d*delta_s3))

    pred_a=101*(a+b+c)

    return pred_a

