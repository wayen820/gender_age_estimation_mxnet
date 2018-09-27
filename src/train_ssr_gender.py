from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import copy
import logging
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import sklearn
import ssr_gender
from ssr_iter_gender import SSR_ITER

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_symbol(args, arg_params, aux_params):
    netType1 = args.netType1
    netType2 = args.netType2
    stage_num = [3,3,3]
    lambda_local = 0.25*(netType1%5)
    lambda_d = 0.25*(netType2%5)

    out = ssr_gender.get_symbol(stage_num,lambda_local,lambda_d)
    y=mx.symbol.Variable('label_gender')
    lro=mx.sym.MAERegressionOutput(out,label=y,name='lro')
    return (lro,arg_params,aux_params)

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='../datasets/imdb', help='training set directory')
  parser.add_argument('--prefix', default='../model/wiki_ssr', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: always save')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='ssr', help='specify network')
  parser.add_argument('--lr', type=float, default=0.0001, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument("--netType1", type=int, default=4,required=True,help="network type 1")
  parser.add_argument("--netType2", type=int, default=4,required=True,help="network type 2")
  args = parser.parse_args()
  return args

def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    if args.per_batch_size==0:
      args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    #args.rescale_threshold = 0
    args.image_channel = 3

    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list)==1
    data_dir = data_dir_list[0]
    # path_imgrec = None
    # path_imglist = None
    args.num_classes = 0
    image_size = (64,64)
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    path_imgrec = os.path.join(data_dir, "train.rec")

    print('Called with argument:', args)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    # if args.network[0]=='s':
    #   data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
    #   spherenet.init_weights(sym, data_shape_dict, args.num_layers)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        data_names=['data'],
        label_names=['label_gender']
    )
    print(data_shape)
    train_dataiter = SSR_ITER(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        mean                 = mean,
    )
    val_rec = os.path.join(data_dir, "val.rec")
    val_iter = None
    if os.path.exists(val_rec):
        val_iter = SSR_ITER(
            batch_size           = args.batch_size,
            data_shape           = data_shape,
            path_imgrec          = val_rec,
            shuffle              = False,
            mean                 = mean,
        )
    print(train_dataiter.provide_label)

    initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    # initializer = mx.init.Xavier(rnd_type='uniform')

    _rescale = 1.0/args.ctx_num
    # opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    opt=optimizer.Adam(learning_rate=base_lr,rescale_grad=_rescale)
    #opt = optimizer.SGD(learning_rate=base_lr)
    #opt = optimizer.Nadam(learning_rate=base_lr, wd=base_wd, rescale_grad=_rescale)
    som = 50
    _cb = mx.callback.Speedometer(args.batch_size, som)

    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [40000, 60000, 80000]
      # if args.loss_type>=1 and args.loss_type<=7:
      #   lr_steps = [100000, 140000, 160000]
      # p = 512.0/args.batch_size
      for l in range(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l])
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        if args.ckpt==0:
          do_save = False
        elif args.ckpt>1:
          do_save = True
        if do_save:
          print('saving', msave)
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        # print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    epoch_cb = None

    a = mx.viz.plot_network(sym, shape={"data": (1, 3, 64, 64)}, node_attrs={"shape": 'rect', "fixedsize": 'false'})
    a.render('xx')

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_iter,
        eval_metric        = 'mae',
        kvstore            = 'device',
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

