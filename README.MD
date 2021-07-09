# [Evolving Neural Architecture Using One Shot Model](https://dl.acm.org/doi/abs/10.1145/3449639.3459275)

**This code is based on the implementation of [DARTS](https://github.com/quark0/darts) and [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects).**
**The training code for the ImageNet dataset is taken from [P-DARTS](https://github.com/chenxin061/pdarts)**


## Requirements
```
Python >= 3.6.9, PyTorch == 1.5.0, torchvision == 0.6.0
```

## Introduction
This repository contains the code for our work called Evolving Neural Architecture Search using One Shot Model (EvNAS). In our method,
the architectures are represented by using the architecture parameter of the one-shot model which results in the weight sharing among
the architectures for a given population of architectures and also weight inheritance from one generation to the next generation of architectures.
We propose a decoding technique for the architecture parameter which is used to divert majority of the gradient information towards the
given architecture and is also used for improving the performance prediction of the given architecture from the one-shot model
during the search process. Furthermore, we use the accuracy of the partially trained architecture on the validation data as a prediction
of its fitness in order to reduce the search time.

## For Search Space S1
### Pretrained models
**CIFAR-10** ([EvNAS_A_cifar_weights.pt](https://drive.google.com/file/d/1NFLOLLBwdq79QB2O4S7lXV4NL6Oho0ba/view?usp=sharing))
```
python3 test_cifar.py --arch EvNASA --model_path EvNAS_A_cifar_weights.pt
```
For EvNAS-B and EvNAS-C use EvNASB and EvNASC respectively for --arch option with their pretrained weigths
([EvNAS_B_cifar_weights.pt](https://drive.google.com/file/d/1zV1idPB0dZ3T8t5TwdCMHhMvvsK-yUW7/view?usp=sharing)),
([EvNAS_C_cifar_weights.pt](https://drive.google.com/file/d/17rU4zO_2l21-OdJLH9ffk-n8_xDFK7-M/view?usp=sharing))<br />
** Expected result for EvNAS-A: 2.40% test error rate (97.60% top-1 accuracy) with 3.6M model params. <br />
** Expected result for EvNAS-B: 2.51% test error rate (97.49% top-1 accuracy) with 3.8M model params.<br />
** Expected result for EvNAS-C: 2.55% test error rate (97.45% top-1 accuracy) with 3.4M model params.<br />

**CIFAR-100** ([EvNAS_A_cifar100_weights.pt](https://drive.google.com/file/d/1xEN9eQAB2pIvufqmQpGo_W3-KB9LIUi1/view?usp=sharing))
```
python3 test_cifar100.py --arch EvNASA --model_path EvNAS_A_cifar_weights.pt
```
For EvNAS-B and EvNAS-C use EvNASB and EvNASC respectively for --arch option with their pretrained weigths
([EvNAS_B_cifar100_weights.pt](https://drive.google.com/file/d/1brjKC4vDUAfbi2QWDewEXVLWxETdVX7g/view?usp=sharing)),
([EvNAS_C_cifar100_weights.pt](https://drive.google.com/file/d/1WuHmwc9n3BakOqHlcUuqN30MRlRx7ziT/view?usp=sharing))<br />
** Expected result for EvNAS-A: 16.05% test error rate (83.95% top-1 accuracy) with 3.6M model params. <br />
** Expected result for EvNAS-B: 16.08% test error rate (83.92% top-1 accuracy) with 3.8M model params.<br />
** Expected result for EvNAS-C: 16.46% test error rate (83.54% top-1 accuracy) with 3.4M model params.<br />

**ImageNet** ([EvNAS_A_imagenet_model_best.pt.tar](https://drive.google.com/file/d/15I5D2vkkGSOemAckxOqDfPCPTPsHeFRR/view?usp=sharing))
```
python test_imagenet.py --arch EvNASA --model_path EvNAS_A_imagenet_model_best.pth.tar
```
For EvNAS-B and EvNAS-C use EvNASB and EvNASC respectively for --arch option with their pretrained weigths
([EvNAS_B_imagenet_model_best.pt.tar](https://drive.google.com/file/d/1fpwY2fpioWFKmIFvC1DIZEamvdbT6ll5/view?usp=sharing))
([EvNAS_C_imagenet_model_best.pt.tar](https://drive.google.com/file/d/1ulQVVOAiHi6f-yst7XChxuT7QybbwB9Z/view?usp=sharing))<br />
** Expected result for EvNAS-A: 24.4% top-1 error (75.6% top-1 accuracy) and 7.4% (92.6% top-5 accuracy) top-5 error with 5.1M model params.<br />
** Expected result for EvNAS-B: 24.4% top-1 error (75.6% top-1 accuracy) and 7.4% (92.6% top-5 accuracy) top-5 error with 5.3M model params.<br />
** Expected result for EvNAS-C: 25.1% top-1 error (74.9% top-1 accuracy) and 7.8% (92.2% top-5 accuracy) top-5 error with 4.9M model params.<br />

### Architecture search (using CIFAR-10 dataset)
```
python -i train_search.py --cutout
```

### Architecture evaluation (using full-sized models)
To evaluate our best cells by training from scratch on CIFAR-10 dataset, run
```
python train.py --cutout --auxiliary --epochs 600 --arch EvNASA	    # when architecture is present in genotype.py
or
python train.py --cutout --auxiliary --epochs 600 --dir search_DIR # when architecture is in search_DIR as genotype.pickle file
```

### Results
#### CIFAR-10 Result
![cifar_table](img/cifar_res.png)
#### ImageNet Result
![imagenet_table](img/imagenet_res.png)

### Searched Cells
#### Normal Cell of EvNAS-A
![normal_cell](img/normal_cell.png)

#### Reduction Cell of EvNAS-A
![reduce_cell](img/reduce_cell.png)

## For Search Space S2
### Dataset
To download ImageNet-16-120 use the [link](https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4). To use the NAS-201
benchmark for evaluating the search method, download the file [NAS-Bench-201-v1_1-096897.pth](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view)

### Architecture search
Using CIFAR-10:
```
 bash ./searchNAS201.sh cifar10 0 outputs
```
Using CIFAR-100:
```
 bash ./searchNAS201.sh cifar100 0 outputs
```
Using ImageNet-16-120:
```
 bash ./searchNAS201.sh ImageNet16-120 0 outputs
```
### Results
![NAS201_table](img/NAS201.png)

## Reference
To cite our paper, please use the following:
```Latex
@inproceedings{sinha2021evolving,
  title={Evolving neural architecture using one shot model},
  author={Sinha, Nilotpal and Chen, Kuan-Wen},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={910--918},
  year={2021}
}
```
