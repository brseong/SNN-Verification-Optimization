export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

encoding="latency"
delta=2
test_type="mnist"
manual_indices=(48540)
# solvers=("np" "z3" "milp")
solver="np"
strategy="--psm"
# adv=""
num_samples=14
hidden_neuron=20
num_steps=5
repeat=1
python batch_test.py -p ${encoding} --delta-max ${delta} --test-type ${test_type} --${solver} ${strategy} --num-samples ${num_samples} --n-hidden-neurons ${hidden_neuron} --num-steps ${num_steps} --repeat ${repeat} #--manual-indices ${manual_indices[@]}  