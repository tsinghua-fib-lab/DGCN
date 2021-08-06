# DGCN
This is the official implementation of our WWW'21 paper:  

Yu Zheng, Chen Gao, Liang Chen, Depeng Jin, Yong Li, **DGCN: Diversified Recommendation with Graph Convolutional Networks**, In Proceedings of the Web Conference 2021.

***
First unzip the datasets and start the visdom server:
```
visdom -port 33333
```

Then simply run the following command to reproduce the experiments on corresponding dataset and model:
```
python app.py --flagfile ./config/xxx.cfg
```

If you use our codes and datasets in your research, please cite:
```
@inproceedings{zheng2021dgcn,
  title={DGCN: Diversified Recommendation with Graph Convolutional Networks},
  author={Zheng, Yu and Gao, Chen and Chen, Liang and Jin, Depeng and Li, Yong},
  booktitle={Proceedings of the Web Conference 2021},
  pages={401--412},
  year={2021}
}
```
