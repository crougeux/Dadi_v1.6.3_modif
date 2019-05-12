#!/bin/bash

for i in `seq 1 1 25`; do
	python script_inference_anneal2_newton_mis_new_models.py -o SI_test -f ../01_Data/indian_26.fs -y IndianL -x IndianB -p 26,36,46 -m SI -z -l -v
((i++))
done

exit 0;
