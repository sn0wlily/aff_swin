CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --master_port 9997 --nproc_per_node=2 main.py --config ./config/expr_rdrop.yaml >> expr_history.out &
