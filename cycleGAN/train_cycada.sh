python train.py --name GTAtoGTA \
    --resize_or_crop=None \
    --loadSize=600 --fineSize=600 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic \
    --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
    --no_flip --batchSize 100 \
    --dataset_mode single --dataroot /home/ubuntu/cycada_release/train_data \
    --which_direction BtoA --no_html
