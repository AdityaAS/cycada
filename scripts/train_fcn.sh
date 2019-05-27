gpu=0
data="singleview_opendr_solid"
datadir='/efs/data/HMR_baby/datasets/'
data2=opendr
model=fcn8s

crop=240
batch=2
iterations=10000
lr=1e-3
momentum=0.99
num_cls=2
# haha
#outdir=results/${data}-${data2}/${model}
outdir=results/${data}/${data}_${model}
mkdir -p results/${data} 

phase='train' 

python scripts/train_fcn.py ${outdir} --phase ${phase} \
	--model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --datadir ${datadir} \
    --dataset ${data} 
