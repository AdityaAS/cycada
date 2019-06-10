# CUDA_VISIBLE_DEVICES=$2 
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=$3 --master_addr="localhost" --master_port=$1 
CUDA_VISIBLE_DEVICES=$1 python train.py --name surr_up3d \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan \
    --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
    --no_flip --batchSize 4 \
    --dataset_mode unaligned --dataroot_A /scratch/users/aditya/cycada_data/surreal_sml --dataroot_B /scratch/users/aditya/up3d_s31 \
    --which_direction BtoA --display_id 1
