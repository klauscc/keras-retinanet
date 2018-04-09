#!/usr/bin/env sh

export PYTHONPATH="`pwd`/cocoApi/:`pwd`:$PYTHONPATH"
DATA_DIR=/ssd_data/users/$USER/004-nsp
mkdir -p $DATA_DIR
if [ ! -L data ]; then
    ln -s "$DATA_DIR" data
fi
