#!/bin/bash

# commande test models demo dadi #
python script_inference_anneal2_newton_mis.py -f ../01_Data/webster_26.fs -y WEBD -x WEBN -p 30,40,50 -m SI,IM,AM,SC,IM2M,AM2M,SC2M -z -l -v
