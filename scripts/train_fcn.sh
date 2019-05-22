gpu=0
data="singleview_opendr_color_100k_copy"
datadir='/home/ubuntu/anthro-efs/anthro-backup-virginia/data/HMR_baby/datasets'
data2=opendr
model=fcn8s


crop=240
#datadir=/x
batch=2
iterations=100000
lr=1e-3
momentum=0.99
num_cls=2

#outdir=results/${data}-${data2}/${model}
outdir=results/${data}/${data}_${model}
mkdir -p results/${data} 


python scripts/train_fcn.py ${outdir} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --datadir ${datadir} \
    --dataset ${data}  #--dataset ${data2} 
