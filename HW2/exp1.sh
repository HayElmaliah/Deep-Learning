#!/bin/bash


copy_results() {

	current_dir=$(pwd)
	target_dir="${current_dir}/results"

	echo "Creating ${target_dir}"
	mkdir -p ${target_dir}

	echo "Moving all jsons files to ${target_dir}"
	mv "${current_dir}"/results/*.json ${target_dir}
}


1_1() {
	# local pool_every="6"
	# local hidden_layers="256 256"

	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "2"
	do
		for hidden_layers in "512"
		do
			for lr in 0.003 #0.00003
			do
				for reg in 0.003 #0.003
				do
					for k in 32 64
					do
						for l in 2 4 8 16
						do
							python3 -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type cnn
						done
					done
					copy_results
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}

1_2() {
	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "1"
	do
		for hidden_layers in "512"
		do
			for lr in 0.003 #0.00003
			do
				for reg in 0.003 #0.003
				do
					for l in 2 4 8
					do
						for k in 32 64 128 256
						do
							python3 -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type cnn
						done
					done
					copy_results
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}

1_3() {
	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "2"
	do
		for hidden_layers in "128" 
		do
			for lr in 0.001 #0.003
			do
				for reg in 0.001 #0.003
				do
					for l in 2 3 4
					do
						k="64 128"
						python3 -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type cnn
					done
					copy_results
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}

1_4() {
	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "2"
	do
		for hidden_layers in "512"
		do
			for lr in 0.004
			do
				for reg in 0.001
				do
					for l in 8 16 32
					do
						k=32
						python3 -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type resnet
					done

					for l in 2 4 8
					do	
						k="64 128 256"
						python3 -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type resnet
					done	

					copy_results
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}

# 1_1

# 1_2

1_3

# 1_4
