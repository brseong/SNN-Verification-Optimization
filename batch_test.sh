export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python batch_test.py -p Control --delta_max 1&\
python batch_test.py -p latency --delta-max 1 --test-type mnist --milp --num-samples 1000