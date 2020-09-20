# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""构建基础的数据读取模块
- 由于大部分的NLP任务其实数据量都不大，而且文本基本上是能够全部加载到内存的，所以我们直接处理文本文件而不是转换为tf-records
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json

import tensorflow as tf



BERT_UNKOWN_TOKEN = "[UNK]"
BERT_CLS_TOKEN = "[CLS]"
BERT_SEP_TOKEN = "[SEP]"
BERT_MASK_TOKEN = "[MASK]"

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class BaseDataset(object):
  def __init__(self, tokenizer, file_path, mode, label_file=None, max_seq_len=512):
    self.file_path = file_path
    self.mode = mode
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.prefetched = False
    self.data = []
    # 如果是测试数据，那么label可以设置一个默认值，预测的时候不会使用
    # 如果是回归任务，label_file可以为空，这里默认数据是float32类型的
    self.label_map = self.get_label_map(label_file)
    self.id2label_map = {v:k for k,v in self.label_map.items()}
    self._data_size = None

  def get_label_map(self, label_file):
    label_map = {}
    if label_file:
      for (i, label) in enumerate(open(label_file).readlines()):
        label_map[label.strip()] = i
    return label_map

  @property
  def label_num(self):
    return len(self.label_map)
    
  def encode_single_sentence(self, sentence, segment=0):
    tokens = list(self.tokenizer.tokenize(sentence))
    # Account for CLS and SEP
    tokens = tokens[0:self.max_seq_len-2]
    tokens = [BERT_CLS_TOKEN] + tokens + [BERT_SEP_TOKEN]
    return self.tokenizer.convert_tokens_to_ids(tokens), [segment] * len(tokens)

  def encode_two_sentences(self, text_a, text_b):
    tokens_a = list(self.tokenizer.tokenize(text_a))
    tokens_b = list(self.tokenizer.tokenize(text_b))
    _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_len - 3)
    tokens = [BERT_CLS_TOKEN] + tokens_a + [BERT_SEP_TOKEN] + tokens_b + [BERT_SEP_TOKEN]
    segments = [0] + [0] * len(tokens_a) + [0] + [1]*len(tokens_b) + [1]
    return self.tokenizer.convert_tokens_to_ids(tokens), segments

  def process_single_example(self, item):
    raise NotImplementedError

  def filter_example(self, item):
    return False

  def make_generator(self, prefetch=True):
    if self.prefetched:
      for item in self.data:
        yield item
    else:
      for line in open(self.file_path):
        item = json.loads(line)
        if self.filter_example(item):
          continue
        try:
          item = self.process_single_example(item)
        except Exception as e:
          logging.warning('Error when parsing {}, with error {}'.format(item,e))
          continue
        if prefetch:
          self.data.append(item)
        yield item
      if prefetch:
        self.prefetched = True

  @property
  def output_shape(self):
    raise NotImplementedError

  @property
  def output_type(self):
    raise NotImplementedError

  @property
  def data_size(self):
    if self._data_size is None:
      #比较耗时的方法，直接跑一轮数据
      logging.warning('没有提前设置数据大小，将会跑一轮数据，速度会比较慢...')
      logging.info('Prefetch data...')
      self._data_size = 0
      for _ in self.make_generator():
        if self._data_size % 1000 == 0:
          logging.info(f'processed {self._data_size} ')
        self._data_size += 1
      logging.info(f'Datasize - {self._data_size}')
    return self._data_size

class NERDataset(BaseDataset):
  '''一般ner的数据格式为
      A\tO
      B\tB-tag
    中文一般都是按字来进行NER，因此我们这里也默认用字符来处理。在有英文的时候可能会有一些问题，丢失了英文的词信息。
  '''
  def process_single_example(self, item):
    text_a = item['text'].lower()
    label = [self.label_map[lb] for lb in item['labels']]
    tokens = list(text_a)
    assert len(label) == len(tokens)
    # Account for CLS and SEP
    tokens = tokens[0:self.max_seq_len-2]
    label = label[0:self.max_seq_len-2]
    tokens = [BERT_CLS_TOKEN] + [t if t in self.tokenizer.vocab else BERT_UNKOWN_TOKEN for t in tokens] + [BERT_SEP_TOKEN]
    label = [0] + label + [0]
    tokens = self.tokenizer.convert_tokens_to_ids(tokens)
    segments = [0] * len(tokens)
    return {'text_a':(tokens,segments),'label':label}

  @property
  def output_shape(self):
    return {
      "text_a":(tf.TensorShape([None]),tf.TensorShape([None])),
      "label": tf.TensorShape([None])
    }

  @property
  def output_type(self):
    return {
      'text_a':(tf.int32,tf.int32),
      'label': tf.int32 
    }


class SentencesDataset(BaseDataset):
  def process_single_example(self, item):
    text_a = item['text_a']
    label = self.label_map[item['label']] if self.label_map else item['label']
    if 'text_b' in item:
      text_b = item['text_b']
      text,segment = self.encode_two_sentences(text_a,text_b)
      return {'text_a':(text,segment),'label':label}
    else:
      text_a, segment_a = self.encode_single_sentence(text_a)
      return {'text_a':(text_a,segment_a),'label':label}
      
  @property
  def output_shape(self):
    return {
      "text_a":(tf.TensorShape([None]),tf.TensorShape([None])),
      "label": tf.TensorShape([])
    }

  @property
  def output_type(self):
    return {
      'text_a':(tf.int32,tf.int32),
      'label': tf.int32 if self.label_map else tf.float32
    }

class ContrastiveDataset(BaseDataset):
  def process_single_example(self, item):
    text_a, segment_a = self.encode_single_sentence(item['text_a'])
    text_b, segment_b = self.encode_single_sentence(item['text_b'])
    label = self.label_map[item['label']] if self.label_map else item['label']
    return {'text_a':(text_a,segment_a),'text_b':(text_b,segment_b),'label':label}

  @property
  def output_shape(self):
    return {
      "text_a": (tf.TensorShape([None]),tf.TensorShape([None])),
      "text_b":  (tf.TensorShape([None]),tf.TensorShape([None])),
      "label": tf.TensorShape([])
    }

  @property
  def output_type(self):
    return {
      'text_a': (tf.int32,tf.int32),
      'text_b': (tf.int32,tf.int32),
      'label': tf.int32 if self.label_map else tf.float32
    }

class TripletDataset(BaseDataset):
  def filter_example(self, item):
    if item['label'] == '0':
      return True
    return False
    
  def process_single_example(self, item):
    text_a, segment_a = self.encode_single_sentence(item['text_a'])
    text_b, segment_b = self.encode_single_sentence(item['text_b'])
    # label暂时没有用到，随便设置了一个值
    return {'text_a':(text_a,segment_a),'text_b':(text_b,segment_b),'label':0}

  @property
  def output_shape(self):
    return {
      "text_a": (tf.TensorShape([None]),tf.TensorShape([None])),
      "text_b":  (tf.TensorShape([None]),tf.TensorShape([None])),
      "label": tf.TensorShape([])
    }

  @property
  def output_type(self):
    return {
      'text_a': (tf.int32,tf.int32),
      'text_b': (tf.int32,tf.int32),
      'label': tf.int32 if self.label_map else tf.float32
    }

class SiameseDataset(BaseDataset):
  def process_single_example(self, item):
    text_a, segment_a = self.encode_single_sentence(item['text_a'])
    if self.mode != 'predict':
      text_b, segment_b = self.encode_single_sentence(item['text_b'],1)
      label = self.label_map[item['label']] if self.label_map else item['label']
      return {'text_a':(text_a,segment_a),'text_b':(text_b,segment_b),'label':label}
    else:
      return {'text_a':(text_a,segment_a)}
    
  @property
  def output_shape(self):
    if self.mode != 'predict':
      return {
        "text_a": (tf.TensorShape([None]),tf.TensorShape([None])),
        "text_b":  (tf.TensorShape([None]),tf.TensorShape([None])),
        "label": tf.TensorShape([])
      }
    else:
      return {
        "text_a":(tf.TensorShape([None]),tf.TensorShape([None]))
      }

  @property
  def output_type(self):
    if self.mode != 'predict':
      return {
        'text_a': (tf.int32,tf.int32),
        'text_b': (tf.int32,tf.int32),
        'label': tf.int32 if self.label_map else tf.float32
      }
    else:
      return {
        'text_a':(tf.int32,tf.int32)
      }
  
  

