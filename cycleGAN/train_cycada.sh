CUDA_VISIBLE_DEVICES=0 python train.py --name  tex_2_solid \
   	--resize_or_crop='resize_and_crop' \
    --loadSize=256 --fineSize=256 --which_model_netD n_layers --n_layers_D 3 \
    --which_model_netG 'unet_256' \
    --model cycle_gan_socher --input_nc 3 --output_nc 3\
    --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
    --display_id 1 --which_model_netM 'fcn8s'\
    --no_flip --batchSize 1 \
    --dataset_mode unaligned_A_labeled --dataroot /home/ubuntu/anthro-efs/anthro-backup-virginia/data/HMR_baby/datasets/cycada_train \
    --which_direction AtoB --no_html --nThreads 8
