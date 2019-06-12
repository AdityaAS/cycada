
model=cycada_svhn2mnist_noIdentity
epoch='latest'

CUDA_VISIBLE_DEVICES=$1 python test.py --name ${model} \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan \
    --no_flip --batchSize 100 \
    --dataset_mode unaligned --dataroot_A /scratch/users/aditya/data/singleview_opendr_real_ratio --dataroot_B /scratch/users/aditya/data/singleview_opendr_solid \
    --which_direction BtoA \
    --phase train \
    --which_epoch ${epoch}

