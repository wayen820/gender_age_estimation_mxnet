#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=../datasets/wiki/

NETWORK=ssr
JOB=wiki
MODELDIR="../models/ssr_wiki_gender_1_1_2"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model-$NETWORK-$JOB"
LOGFILE="$MODELDIR/log_ssr_wiki"

PRETRAINED="../models/ssr_imdb_gender_1_1/model-ssr-imdb,20"

CUDA_VISIBLE_DEVICES='0' python3 -u train_ssr_gender.py --ckpt 2 --data-dir $DATA_DIR --network "$NETWORK" --wd 0.00004 --end-epoch 100 --pretrained "$PRETRAINED" --prefix "$PREFIX" --per-batch-size 50 --lr 0.0001 --lr-steps 10000,20000,30000 --netType1 4 --netType2 4 > "$LOGFILE" 2>&1 &

