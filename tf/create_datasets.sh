#!/bin/bash

export CUDA_VISIBLE_DEVICES=''

folder_src='src'
folder_in='..'
folder_out='.'

for name in 'shapes' 'mnist' 'dsprites' 'abstract' 'clevr' 'shop' 'gso'; do
    python $folder_src'/convert_tfrecord.py' \
        --folder_in $folder_in \
        --folder_out $folder_out \
        --name $name
done
