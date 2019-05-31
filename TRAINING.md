# Guide on training cycada for a given Source-Target pair distributions

## Background Information
* What dataloader is used for what? (ex: OpenDR dataloader for opendr data formats, Unpaired for cycleGAN stuff etc.) - have Sohan write this
* Where are models saved and loaded from - Have Sohan write this
* Other relevant information relevant to this code base.

## Pretrain FS (Source task network)
* SEGMENTATION: Run train_fcn.sh (point it to directory containing images and segmasks)

## Run CycleGAN and get source translated images
* SEGMENTATION: Run CycleGAN (train_cycada.sh) with semantic consistency loss (this loss is specific to cycada)

## Finetuning target network (Feature wise adaptation?)
* SEGMENTATION: 
