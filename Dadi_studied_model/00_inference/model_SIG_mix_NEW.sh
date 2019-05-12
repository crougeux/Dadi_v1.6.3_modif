#!/bin/bash

for i in `seq 1 1 1`; do
	python script_inference_anneal2_newton_mis_new_models.py -o  -f ../01_Data/ -y  -x  -p 26,36,46 -m SIG -z -l -v
((i++))
done

exit 0;
