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
snapshot=5
batch=1

weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='color2blk'
tgt='blk'
datadir='/home/users/aditya/data'

resdir="results/${src}_to_${tgt}/adda_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
# model='drn26'
# baseiter=115000
model='fcn8s'
baseiter=100000

num_cls=2

base_model="/home/users/aditya/sohan/cycada/runs/fcn8s/singleview_opendr_solid/color/checkpoints/iter49.pth"
outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"

# mkdir -p outdir

# Run python script #
python scripts/train_fcn_adda.py \
	--output ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --model ${model} --weights_init ${base_model}\
    --weights_shared --discrim_feat --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot ${snapshot} --num_cls ${num_cls}
