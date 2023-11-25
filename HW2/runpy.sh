#!bin/bash
python -m hw2.experiments run-exp -n "exp${exp_name}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --model-type resnet
