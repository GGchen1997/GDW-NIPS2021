# Generalized Data Weighting via Class-level Gradient Manipulation

This repository is the official implementation of [Generalized Data Weighting via Class-level Gradient Manipulation (NeurIPS 2021)](http://arxiv.org/abs/2111.00056). 

<div  align="center"> 
<img src="./pic/intro.png" width = "400" height = "250" align=center />
</div>
<br/><br/>

If you find this code useful in your research then please cite:   
```bash
@inproceedings{
chen2022bidirectional,
title={Bidirectional Learning for Offline Infinite-width Model-based Optimization},
author={Can Chen and Yingxue Zhang and Jie Fu and Xue Liu and Mark Coates},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=_j8yVIyp27Q}
}
``` 



## Requirements

- Linux
- Python 3.7
- Pytorch 1.9.0
- Torchvision 0.9.1

More specifically, run this command:

```setup
pip install -r requirements.txt
```

## Run mw-net and gdw on CIFAR10

Download [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and place it in *./data*.

To compare mw-net and gdw on CIFAR10 under 40% uniform noise, run this command:
```train
python -u  main.py --corruption_prob 0.4 --dataset cifar10 --mode mw-net --outer_lr 100
python -u  main.py --corruption_prob 0.4 --dataset cifar10 --mode gdw --outer_lr 100
```
We set the outer level learning as 100 on CIFAR10 and 1000 on CIFAR100.

## Results
We place training logs of the above command in *./log* and list results as below:

| Method         | mw-net  | gdw |
| ------------------ |---------------- | -------------- |
| Accuracy   |     86.62%         |      87.97%       |


## Acknowledgements
We thank the Pytorch implementation on mw-net(https://github.com/xjtushujun/meta-weight-net).
