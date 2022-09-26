export OMP_NUM_THREADS=16

torchrun --nproc_per_node=2 --master_port=19198 big_demo.py
