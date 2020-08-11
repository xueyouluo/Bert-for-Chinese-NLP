# Bert-for-Chinese-NLP

Bert for Chinese NLP tasks, developed with TensorFlow 2.x.

> 这个项目主要是为了用来学习tf2.x的，因为本人是tf1.x的遗老，但是还是不想迁移到pytorch，所以就升级到tf2.x吧。

## 准备

### 依赖

本项目有一些依赖，需要手动配置，主要包括：

- tensorflow-gpu==2.3.0
- tensorflow的models项目
  - 项目地址：[models](https://github.com/tensorflow/models/tree/master/official)
  - 安装方法有两种，一种是直接clone source（我使用的是这个方法），一种是直接pip安装（未测试过）
  > clone tensorflow的代码的话里面有一处需要更改，即official/modeling/activation/gelu.py的实现，目前tf2.3还没有支持这个最新的api，需要修改为[原始的实现方法](https://github.com/tensorflow/models/blob/v2.3.0/official/modeling/activations/gelu.py#L27)。

  > 当然也可以直接clone我更新后的[Models仓库](https://github.com/xueyouluo/models)

### Bert预训练模型转换

由于这里的bert是使用tf2.x实现的，因此原来开源的bert模型的checkpoint都不能直接使用，需要转换为tf2.x版本的才可以使用。

这里使用了models里面提供的convert脚本直接进行转换，具体可以参考bash/convert_bert_checkpoint.sh

> 需要注意的是转换后的checkpoint会以-1结尾，这应该是step，需要注意在使用的时候加入完整的checkpoint name，参考scripts/train_classifier.sh的init_checkpoint参数。

## 说明和使用

### 分类任务

#### 数据准备

训练数据需要以jsonl的形式保存，即每一行是一个json.dumps的数据。每个json数据需要包括以下字段：
- text_a: 输入文本
- text_b: 另外一个输入文本（可选），一般只在存在两个句子的时候需要用到
- label: 如果是分类任务是具体的label名称，如果是回归任务则是具体的数字

如果是分类任务的话还需要额外提供一个label文件，每一行是一个label名称，与训练数据的label名称一致，具体可以参考labels文件夹下的文件。如果是回归任务则不需要提供。

#### 传统BERT

参考bash/train_classifier.sh的设置。

最好提前设置好train_data_size、eval_data_size，这样就不需要运行一轮数据处理代码获取数据集大小。当然数据集较大的时候，你也可以自己设置一个值，配合`num_train_epochs`来控制整个训练步数：`training steps = train_data_size * num_train_epochs`。

其他可以配置的参数需要参考models代码库中bert项目下的common_flags.py中的信息。

#### Siamese-BERT

参考了文章《Sentence-BERT-Sentence Embeddings using Siamese BERT-Networks》的实现，可以用于做sentence embedding。

与传统的bert训练一样，只是model_type需要设置为`siamese`。

> 在中文lcqmc数据集和英文的MRPC、QQP等数据集上做过测试，能够跑通，但是evaluation的loss基本不怎么降，accuracy与传统的bert也有较大的差距，可能是有bug待排查。

> 如果是句子间的关系判断还是推荐使用传统的bert，效果好很多。

