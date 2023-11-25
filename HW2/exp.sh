#!/bin/sh


run_experiment() {
		local exp_name=${1}
		local k=${2}
		local l=${3}
		local pool_every=${4}
		local hidden_layers=${5}

		echo "parameters ${exp_name} k=${k} l=${l} P=${pool_every} H=${hidden_layers}"

		# srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp ${params}
		#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n "exp${exp_name}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --model-type resnet
		sbatch -c 2 --gres=gpu:1 -J allexp -o out/allexp_a1 python -m hw2.experiments run-exp -n "exp${exp_name}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --model-type resnet
}

copy_results() {

	current_dir=$(pwd)
	RUN_FULL_NAME=$1
	target_dir="${current_dir}/results/${RUN_FULL_NAME}"

	echo "Creating ${target_dir}"
	mkdir -p ${target_dir}

	echo "Moving all jsons files to ${target_dir}"
	mv "${current_dir}"/results/*.json ${target_dir}
}


1_1() {
	# local pool_every="6"
	# local hidden_layers="256 256"

	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "9"
	do
		for hidden_layers in "256 256 256 256"
		do
			for lr in 0.003 #0.00003
			do
				for reg in 0.03 #0.003
				do
					for k in 32 64
					do
						for l in 2 4 8 16
						do
							sbatch -c 2 --gres=gpu:1 -J allexp -o out/allexp_a1.1 python -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type cnn
						done
					done
					copy_results "exp${FUNCNAME[0]}_P${pool_every}_H${hidden_layers// /-}_lr${lr}_reg${reg}"
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}

1_2() {
	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "9"
	do
		for hidden_layers in "256 256 256 256"
		do
			for lr in 0.003 #0.00003
			do
				for reg in 0.03 #0.003
				do
					for l in 2 4 8
					do
						for k in 32 64 128 256
						do
							sbatch -c 2 --gres=gpu:1 -J allexp -o out/allexp_a1.2 python -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type cnn
						done
					done
					copy_results "exp${FUNCNAME[0]}_P${pool_every}_H${hidden_layers// /-}_lr${lr}_reg${reg}"
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}

1_3() {
	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "6"
	do
		for hidden_layers in "256 256 256 256" 
		do
			for lr in 0.003 #0.00003
			do
				for reg in 0.03 #0.003
				do
					for l in 1 2 3 4
					do
						k="64 128 256"
						sbatch -c 2 --gres=gpu:1 -J allexp -o out/allexp_a1.3 python -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type cnn
					done
					copy_results "exp${FUNCNAME[0]}_P${pool_every}_H${hidden_layers// /-}_lr${lr}_reg${reg}"
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}

1_4() {
	echo "Starting experiment ${FUNCNAME[0]}"
	for pool_every in "6" "9" "12"
	do
		for hidden_layers in "256 256 256 256"
		do
			for lr in 0.003
			do
				for reg in 0.03 0.003
				do
					for l in 8 16 32
					do
						k=32
						sbatch -c 2 --gres=gpu:1 -J allexp -o out/allexp_a1.41 python -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type resnet
					done

					for l in 2 4 8
					do	
						k="64 128 256"
						sbatch -c 2 --gres=gpu:1 -J allexp -o out/allexp_a1.42 python -m hw2.experiments run-exp -n "exp${FUNCNAME[0]}" -K ${k} -L ${l} -P ${pool_every} -H ${hidden_layers} --lr ${lr} --reg ${reg} --model-type resnet
					done	

					copy_results "exp${FUNCNAME[0]}_P${pool_every}_H${hidden_layers// /-}_lr${lr}_reg${reg}"
				done
			done
		done
	done

	echo "Ending experiment ${FUNCNAME[0]}"
}


1_1

1_2

1_3

1_4
