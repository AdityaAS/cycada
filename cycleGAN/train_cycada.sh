# CUDA_VISIBLE_DEVICES=$2 
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=$3 --master_addr="localhost" --master_port=$1 
CUDA_VISIBLE_DEVICES=$1 python train.py --name $3 \
    --resize_or_crop='crop' \
    --which_model_netD n_layers --n_layers_D 3 \
    --no_flip --batchSize 16 \
    --dataroot_A /scratch/users/aditya/surreal_sml --dataroot_B /scratch/users/aditya/up3d_s31 \
    --display_port $2 --nThreads 80