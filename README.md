# Cycle Consistent Adversarial Domain Adaptation (CyCADA)
A [pytorch](http://pytorch.org/) implementation of [CyCADA](https://arxiv.org/pdf/1711.03213.pdf). 


## Setup
* Install python requirements
    * pip install -r requirements.txt

## Running Cycada
* Run the Cycle-GAN code to produce stylized images or download them from here
  * [SVHN as MNIST](https://people.eecs.berkeley.edu/~jhoffman/cycada/svhn2mnist.zip) (114MB)
  * [MNIST as USPS](https://people.eecs.berkeley.edu/~jhoffman/cycada/mnist2usps.zip) (6MB)
  * [USPS as MNIST](https://people.eecs.berkeley.edu/~jhoffman/cycada/usps2mnist.zip) (3MB)
  * Download [GTA as CityScapes](http://efrosgans.eecs.berkeley.edu/cyclegta/cyclegta.zip) images (16GB).
* Place them in folder any folder separately say $stylizedPATH
* Run the code in scripts/train_adda.py as 

```
python scripts/train_adda.py --dd=$stylizedPATH --mn=$modelName --s=$sourceDatasetName --t=targetDatasetName

python scripts/train_adda.py --help                              
usage: train_adda.py [-h] [--s SRC] [--t TGT] [--b BATCHSIZE]
                     [--wd WEIGHT_DECAY] [--dd DATADIR] [--mn MODELNAME]
                     [--m MODEL] [--nc NUMCLASSES] [--pe PIXLEVEPOCHS]
                     [--fe FEATLEVEPOCHS] [--plr PIXLR] [--flr FEATLR]
                     [--iter ITER] [--ns NUMSAVE]

optional arguments:
  -h, --help          show this help message and exit
  --s SRC
  --t TGT
  --b BATCHSIZE
  --wd WEIGHT_DECAY
  --dd DATADIR
  --mn MODELNAME
  --m MODEL
  --nc NUMCLASSES
  --pe PIXLEVEPOCHS
  --fe FEATLEVEPOCHS
  --plr PIXLR
  --flr FEATLR
  --iter ITER
  --ns NUMSAVE
```

<!-- ## Train image adaptation only (digits)
* Image adaptation builds on the work on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The submodule in this repo is a fork which also includes the semantic consistency loss. 
* Pre-trained image results for digits may be downloaded here
  * [SVHN as MNIST](https://people.eecs.berkeley.edu/~jhoffman/cycada/svhn2mnist.zip) (114MB)
  * [MNIST as USPS](https://people.eecs.berkeley.edu/~jhoffman/cycada/mnist2usps.zip) (6MB)
  * [USPS as MNIST](https://people.eecs.berkeley.edu/~jhoffman/cycada/usps2mnist.zip) (3MB)
* Producing SVHN as MNIST 
   * For an example of how to train image adaptation on SVHN->MNIST, see `cyclegan/train_cycada.sh`. From inside the `cyclegan` subfolder run `train_cycada.sh`. 
   * The snapshots will be stored in `cyclegan/cycada_svhn2mnist_noIdentity`. Inside `test_cycada.sh` set the epoch value to the epoch you wish to use and then run the script to generate 50 transformed images (to preview quickly) or run `test_cycada.sh all` to generate the full ~73K SVHN images as MNIST digits. 
   * Results are stored inside `cyclegan/results/cycada_svhn2mnist_noIdentity/train_75/images`. 
   * Note we use a dataset of mnist_svhn and for this experiment run in the reverse direction (BtoA), so the source (SVHN) images translated to look like MNIST digits will be stored as `[label]_[imageId]_fake_B.png`. Hence when images from this directory will be loaded later we will only images which match that naming convention.
--!>

## Train Feature Adaptation for Semantic Segmentation
* Download [GTA DRN-26 model](https://people.eecs.berkeley.edu/~jhoffman/cycada/drn26-gta5-iter115000.pth)
* Download [GTA as CityScapes DRN-26 model](https://people.eecs.berkeley.edu/~jhoffman/cycada/drn26-cyclegta5-iter115000.pth)
* Adapt using `scripts/train_fcn_adda.sh`
   * Choose the desired `src` and `tgt` and `datadir`. Make sure to download the corresponding base model and data. 
   * The final DRN-26 CyCADA model from GTA to CityScapes can be downloaded [here](https://people.eecs.berkeley.edu/~jhoffman/cycada/drn26_cycada_cyclegta2cityscapes.pth)
