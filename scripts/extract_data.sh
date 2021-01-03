#!/bin/bash

target_dir=./generated/tmp

mkdir -p $target_dir

for d in $(ls -d ./data/*)
do
    unzip $d -d $target_dir
done
