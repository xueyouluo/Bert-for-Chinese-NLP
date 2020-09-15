# Bert-for-Chinese-NLP

Bert for Chinese NLP tasks, developed with TensorFlow 2.x.

> 这个项目主要是为了用来学习tf2.x的，因为本人是tf1.x的遗老，但是还是不想迁移到pytorch，所以就升级到tf2.x吧。

> ***大部分任务未经过详细的测试，后期慢慢补实验结果。***

## 准备

### 依赖

本项目有一些依赖，需要手动配置，主要包括：

- python==3.6
- tensorflow-gpu==2.3.0
- tensorflow的models项目
  - 项目地址：[models](https://github.com/tensorflow/models/tree/master/official)
  - 安装方法有两种：
    - 一种是直接clone source（我使用的是这个方法）
      > clone tensorflow的代码的话里面有一处需要更改，即official/modeling/activation/gelu.py的实现，目前tf2.3还没有支持这个最新的api，需要修改为[原始的实现方法](https://github.com/tensorflow/models/blob/v2.3.0/official/modeling/activations/gelu.py#L27)。
      
      > 当然也可以直接clone我更新后的[Models仓库](https://github.com/xueyouluo/models)
    - 一种是直接pip安装:`pip install tf-models-official`
      > 不过不太推荐，本人测试过，有两个问题：它会安装tensorflow==2.3.0，但是我们一般是需要用tensorflow-gpu==2.3.0的版本，另外即使卸载了tf-2.3重新安装tf-gpu-2.3，仍然会出现reshape的报错。我没有深入的去debug了。

### Bert预训练模型转换

由于这里的bert是使用tf2.x实现的，因此原来开源的bert模型的checkpoint都不能直接使用，需要转换为tf2.x版本的才可以使用。

这里使用了models里面提供的convert脚本直接进行转换，具体可以参考bash/convert_bert_checkpoint.sh

> 需要注意的是转换后的checkpoint会以-1结尾，这应该是step，在使用的时候加入完整的checkpoint name，参考scripts/train_classifier.sh的init_checkpoint参数。

## 说明和使用

### 分类任务

#### 数据准备

训练数据需要以jsonl的形式保存，即每一行是一个json.dumps的数据。每个json数据需要包括以下字段：
- text_a: 输入文本
- text_b: 另外一个输入文本（可选），一般只在存在两个句子的时候需要用到
- label: 如果是分类任务是具体的label名称，如果是回归任务则是具体的数字

如果是分类任务的话还需要额外提供一个label文件，每一行是一个label名称，与训练数据的label名称一致，具体可以参考labels文件夹下的文件。如果是回归任务则不需要提供。

示例数据可以参考toy_data/lcqmc.json文件。

#### 传统BERT

参考bash/train_classifier.sh的设置。

最好提前设置好train_data_size、eval_data_size，这样就不需要运行一轮数据处理代码获取数据集大小。当然数据集较大的时候，你也可以自己设置一个值，配合`num_train_epochs`来控制整个训练步数：`training steps = train_data_size * num_train_epochs`。

其他可以配置的参数需要参考models代码库中bert项目下的common_flags.py中的信息。

#### Siamese-BERT

> 这部分还是实验性质的代码，没有取得较好的结果

参考了文章《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》的实现，可以用于做sentence embedding。

与传统的bert训练一样，只是model_type需要设置为`siamese`。

##### 分类任务

设置siamese_type为`classify`，这个主要是参考前面的论文用来做预训练的，可以使用NLI的数据跑训练

> 在中文lcqmc数据集和英文的MRPC、QQP等数据集上做过测试，能够跑通，但是evaluation的loss基本不怎么降，accuracy与传统的bert也有较大的差距，***可能是有bug待排查***。

> 如果是句子间的关系判断还是推荐使用传统的bert，效果好很多。但SBERT的目的并不是为了获得SOTA的效果，主要还是为了大量文本的相似度计算，如果效果不错的话可以考虑。

##### Triplet Loss

基本的相似度的loss，使用了欧式距离，参考了[triplet loss](https://www.tensorflow.org/addons/tutorials/losses_triplet)。设置siamese_type为`triplet`。

这个loss一般用在人脸识别上，虽然也可以用在文本上，大部分情况下是要求训练数据是有类别信息的，但是感觉其实直接训练分类模型可能效果还更好一些。所以可能的应用场景就是我们只有一些相似和不相似的平行语料，但是我们不知道它们可以被分成多少类别，因此希望相似的数据尽可能在一起，这可以用于搜索的排序，比如query和doc的相似性判断。如果单纯的两两判断那其实直接用bert就好了，但是如果数据量特别大的时候，我们希望召回一些数据，那么两两判断就不太可行，通过feature embedding然后再利用一些ANN的框架帮助快速找到相似的内容也许更靠谱一些。

我在lcqmc的数据集上测试了一下，只选取了数据集中label为1的作为训练数据，这只解决了anchor和positive的问题，negative我使用了batch hard的方法，即只找同一个batch中其他最相似的postive作为negative，当然也可以用其他的方法提前找到negative。

> evaluation的时候使用了同样的方法来计算metric，即在同一个batch中，anchor对应的postive之间的距离应该是最小的。

在lcqmc的简单测试结果，batch size=128：

> `loss: 0.0874 - accuracy: 99.1379 - val_loss: 0.1809 - val_accuracy: 97.7902`

##### Contrastive Loss

这个参考了[tfa](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/contrastive_loss)的实现。与triplet loss类似，也是为了获取更好的feature embedding，最好也是提供有标签的数据。

使用了lcqmc的数据集进行了测试，这里可以用label为1和0的两类数据集了，evaluation的时候使用了margin作为分类标准，认为两个数据之间的距离小于margin就是相似的来计算准确率，但是观察平均的负例之间的距离很难达到margin，于是设置了margin/2，但是实验发现也是没有太大的提升。下面是使用lstm作为encoder得到的baseline结果（margin=1.0）：

> `loss: 0.1506 - accuracy: 17.1214 - postive_distance: 0.3287 - negative_distance: 0.6713 - val_loss: 0.2430 - val_accuracy: 64.2344 - val_postive_distance: 0.2729 - val_negative_distance: 0.4384`

可以发现测试集上负例之间的平均距离也是小于0.5的，所以这个margin还是挺难调节的。暂时没有时间去继续深入研究。

> 使用bert发现正例之间以及负例之间的距离非常接近，需要进一步debug一下。

##### Additive Margin Softmax

最近读到了Google的论文《Language-agnostic BERT Sentence Embedding》，其中提到的additive margin softmax挺有趣的，因此也实现了一下（其实就是AM-Softmax了，只是稍微有点不同而已），将simese_type设置为ams即可。

一开始使用LSTM的模型进行了测试，发现效果不行，因此直接改成用bert进行测试，效果开始好了很多。

测试时发现evaluation的loss和accuracy基本没什么变化，而且accuracy效果较差，因此将evaluation的batch size调整到跟训练的大小一样，发现效果好了很多。也正如论文中提到的，这种任务需要batch size越大越好，有机会会研究一下论文中提到的Cross-Accelerator Negative Sampling机制，可以将negative变得很大。

> 但是evalutation的loss还是没有什么变化，把测试数据换成训练数据发现还是能降的，说明模型应该是过拟合到训练数据了，所以还是需要将batch size弄大一些。但是把batch size调整成128后，loss降的还是很慢，而且accuracy居然越训练越低-_-!!这又是个bug？

简单跑了一下lcqmc的任务，结果大概如下，与triplet lose采用了相同的配置，margin设置为0.3：

> `loss: 0.4346 - forward_accuracy: 99.5494 - backward_accuracy: 99.5841 - val_loss: 1.2178 - val_forward_accuracy: 97.8571 - val_backward_accuracy: 97.7679`


google也放出了他们在100多个语言上训练的LaBSE模型，有时间的话把它拿过来在中文上测试一下。

##### 实际检验

从上面的简单的结果来看，triplet loss的效果最好，AMS的效果其次，但是我们需要在进一步实践中检验一下他们的效果，看看哪种方法能够从数据集中找到最相似的句子。

具体参考[SBert_results](./sbert_results.md)。