gpu=0
data="cityscapes"
datadir='/scratch/users/aditya/'
model=resnet

crop=256
batch=16
iterations=10000
lr=1e-3
momentum=0.99
num_cls=2

outdir=results/${data}/${data}_${model}
mkdir -p results/${data} 

phase='train' 

CUDA_VISIBLE_DEVICES=$1 python scripts/train_fcn.py ${outdir} --phase ${phase} \
	--model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --datadir ${datadir} \
    --dataset ${data} 
