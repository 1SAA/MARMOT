export MY_NRPOC=${1:-1}
MY_THREADS=$[128/${MY_NRPOC}]
export OMP_NUM_THREADS=${MY_THREADS}
torchrun --nproc_per_node=${MY_NRPOC} --master_port=19198 opt_el.py