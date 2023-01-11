# Knowledge-Distillation
Pytorch implementation of various Knowledge Distillation (KD) methods. 


## Lists
  Name | Method | Paper Link | Code Link
  :---- | ----- | :----: | :----:
  Baseline | basic model with softmax loss | — | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/train_base.py)
  ST       | soft target | [paper](https://arxiv.org/pdf/1503.02531.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/st.py)
  AT       | attention transfer | [paper](https://arxiv.org/pdf/1612.03928.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py)
  Fitnet   | hints for thin deep nets | [paper](https://arxiv.org/pdf/1412.6550.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/fitnet.py)
  NST      | neural selective transfer | [paper](https://arxiv.org/pdf/1707.01219.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/nst.py)
  FT       | factor transfer | [paper](http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ft.py)
  RKD      | relational knowledge distillation | [paper](https://arxiv.org/pdf/1904.05068.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/rkd.py)
- Note, there are some differences between this repository and the original papers：
	- For `AT`: I use the sum of absolute values with power p=2 as the attention.
	- For `Fitnet`: The training procedure is one stage without hint layer.
	- For `NST`: I employ polynomial kernel with d=2 and c=0.
	
## Datasets
- CIFAR10
- CIFAR100

## Networks
- Resnet-20
- Resnet-110


## Training
- Create `./dataset` directory and download CIFAR10/CIFAR100 in it.
- You can simply specify the hyper-parameters listed in `train_xxx.py` or manually change them.
	- Use `train_base.py` to train the teacher model in KD and then save the model.
	- Before traning, you can choose the method you need in `./kd_losses` directory, and run `train_kd.py` to train the student model.
	

## Requirements
- python 3.7
- pytorch 1.3.1
- torchvision 0.4.2

## Acknowledgements
This repo is partly based on the following repos, thank the authors a lot.
- [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)
- [bhheo/BSS_distillation](https://github.com/bhheo/BSS_distillation)
- [clovaai/overhaul-distillation](https://github.com/clovaai/overhaul-distillation)
- [passalis/probabilistic_kt](https://github.com/passalis/probabilistic_kt)
- [lenscloth/RKD](https://github.com/lenscloth/RKD)
- [AberHu/Knowledge-Distillation-Zoo]（https://github.com/AberHu/Knowledge-Distillation-Zoo）

If you employ the listed KD methods in your research, please cite the corresponding papers.
