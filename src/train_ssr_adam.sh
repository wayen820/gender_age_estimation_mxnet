#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=../datasets/megaage/

NETWORK=ssr
JOB=imdb
MODELDIR="../models/ssr2_megaage_1_1"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model-$NETWORK-$JOB"
LOGFILE="$MODELDIR/log_ssr_imdb"

PRETRAINED="../models/ssr2_wiki_1_1/model-ssr-imdb,15"

CUDA_VISIBLE_DEVICES='0' python3 -u train_ssr.py --ckpt 2 --data-dir $DATA_DIR --network "$NETWORK" --wd 0.00004 --pretrained "$PRETRAINED" --prefix "$PREFIX" --per-batch-size 50 --lr 0.001 --lr-steps 40000,70000,100000 --netType1 4 --netType2 4 --max-steps 120000 > "$LOGFILE" 2>&1 &

