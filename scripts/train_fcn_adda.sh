######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
max_iter=100000
crop=120
snapshot=5000
batch=1

weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='tex2solid'
#'singleview_opendr_solid'
tgt='singleview_opendr_real_ratio'
datadir='/scratch/users/aditya/data'
#'/home/users/aditya/data'

resdir="results/${src}_to_${tgt}/adda_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
# model='drn26'
# baseiter=115000
model='fcn8s'
baseiter=100000

num_cls=2

base_model="results/tex2solid_to_singleview_opendr_real_ratio/adda_sgd/weights_shared_nolsgan_discrim_score/fcn8s/lr1e-5_crop120_ld1_lg0.1_momentum0.99/net-itercurr.pth"
# base_model="/scratch/users/aditya/checkpoints_all/cycada/runs/fcn8s/singleview_opendr_solid/color/checkpoints/iter49.pth"
# base_model="/scratch/users/aditya/checkpoints_all/cycada/runs/fcn8s/singleview_opendr_solid/v5/checkpoints/iter49.pth"
outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"

mkdir -p $outdir

# Run python script #
python scripts/train_fcn_adda.py \
	--output ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --model ${model} --weights_init=$base_model \
    --weights_shared --discrim_feat --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot ${snapshot} --num_cls ${num_cls}
