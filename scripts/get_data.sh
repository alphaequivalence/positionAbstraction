#!/bin/bash

data_dir=./data/

lst=(
    "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_torso.zip"
    "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_bag.zip"
    "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_hips.zip"
    "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-train_hand.zip"
    "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-validation.zip"
    "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-test.zip"
)

echo "[SHL Challenge] Downloading the shldataset. Depending on your bandwidth, this may take a little while."

for l in "${lst[@]}"
do
    echo "[SHL Challenge] downloading from " $l " into " $data_dir " ..."
    wget $l -N -P $data_dir/  # -N overwrite only if there is a new version in the server
done

if [ $? == "0" ]
then
    echo "[SHL Challenge] OK"
else
    echo "[SHL Challenge] Fatal Error"
fi
