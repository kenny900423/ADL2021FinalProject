#!/usr/bin/env bash

if [ ! -d "./BERTNLG/model" ]; then
    (
        cd "./BERTNLG"
        wget https://www.dropbox.com/s/q0rhqw7jon3u65p/nlg.zip?dl=1 -O nlg.zip
        unzip nlg.zip
        rm nlg.zip
    )
fi

if [ ! -d "./T5DST/model" ]; then
    (
        cd "./T5DST"
        wget https://www.dropbox.com/s/naah5bl4hobqgeg/dst.zip?dl=1 -O dst.zip
        unzip dst.zip
        rm dst.zip
    )
fi