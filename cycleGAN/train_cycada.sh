python -m pdb train.py --name cycada_svhn2mnist_noIdentity \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic \
    --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
    --no_flip --batchSize 100 \
    --dataset_mode mnist_svhn --dataroot /efs/data/aditya/cycada/ \
    --which_direction BtoA


# python -m pdb train.py --name cycada_gta2city_noIdentity \
#     --resize_or_crop=None \
#     --loadSize=400 --fineSize=400 --which_model_netD n_layers --n_layers_D 3 \
#     --model cycle_gan \
#     --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
#     --no_flip --batchSize 4 \
#     --dataset_mode unaligned --dataroot /efs/data/aditya/cycada/ \
#     --which_direction BtoA
