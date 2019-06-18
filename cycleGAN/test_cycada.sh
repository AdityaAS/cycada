
model=cycleGANSSIMS-Tr
epoch='latest'

    # --resize_or_crop='resize' \
CUDA_VISIBLE_DEVICES=$1 python test.py --name ${model} \
    --resize_or_crop=None \
    --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan \
    --no_flip --batchSize 100 \
    --dataset_mode unaligned_test --dataroot_A /scratch/users/aditya/surreal_sml --dataroot_B /scratch/users/aditya/up3d_s31 \
    --which_direction AtoB \
    --phase train \
    --which_epoch ${epoch} --nThreads=0

