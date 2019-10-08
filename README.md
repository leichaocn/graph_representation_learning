# 图表示学习

这是《图表示学习入门》系列的配套代码

参考了 [GraphSAGE](https://github.com/williamleif/GraphSAGE)，做了必要的简化，并增加了密集的注释。

## 内容
- Node2Vec
- GraphSAGE（有监督方式）

## 运行
- 运行node2vec代码，请在node2vec文件夹下，运行：
```shell
python node2vec.py
```

- 运行GraphSAGE代码，请在GraphSAGE文件夹下，运行：

```shell
python model.py
```


## 需要软件包
- numpy
- scipy
- networksx
- gensim
- pytorch
- scikit-learn

## 参考
[1] Jure Leskovec, 《Graph Representation Learning》

[2] Jure Leskovec, 《Representation Learning on Networks》 http://snap.stanford.edu/proj/embeddings-www/

[3] https://github.com/williamleif/GraphSAGE
