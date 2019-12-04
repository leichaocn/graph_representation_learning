# 图表示学习

这是《图表示学习入门》系列的配套代码
参考了 [GraphSAGE](https://github.com/williamleif/graphsage-simple)，进行了必要的简化，增加了密集的注释。

## 代码内容

- Node2Vec（在最核心采样方法上做了简化，可能存在问题，建议参考[原版](https://github.com/aditya-grover/node2vec/tree/master/src)）
- GraphSAGE（有监督方式）

## 数据文件

cora文件夹下放着两个文件，均为机器学习领域的2708篇论文的数据（[来源](https://github.com/williamleif/graphsage-simple)）。

- cora.content
  论文词表特征及标签表，用于有监督训练的场景（GraphSAGE）。共2708行，共1435列。
  - 第一列是每个论文的id
  - 中间1433个列是一个词表的统计（表示1433个关键词的有无，如果有是1，没有是0）
  - 最后一列是每个论文的label（即7个子领域）。
- cora.cites 
  论文之间的引用数据表,是图（graph）结构数据。每行对应一个引用关系，有两列：
  - 左列（第一列）是被引用的论文id
  - 右列（第二列）是引用论文的id

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
