# 将models的路径加入到PYTHONPATH环境变量中
export PYTHONPATH="$PYTHONPATH:/nfs/users/xueyou/github/models"
# 不让tf直接霸占全部的gpu显存
export TF_FORCE_GPU_ALLOW_GROWTH=true
export BERT_DIR=/data/xueyou/data/bert_pretrain/converted/chinese_L-12_H-768_A-12
export DATA_DIR=/nfs/users/xueyou/data/chinese_nlp/clue_ner/processed

python run_ner.py \
  --mode='train_and_eval' \
  --use_crf=True \
  --train_data_path=${DATA_DIR}/train.json \
  --eval_data_path=${DATA_DIR}/dev.json \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --max_seq_length=64 \
  --learning_rate=2e-5 \
  --vocab_file=${BERT_DIR}/vocab.txt \
  --label_file=${DATA_DIR}/ner_labels.txt \
  --bert_config_file=${BERT_DIR}/bert_config.json \
  --num_train_epochs=5 \
  --num_eval_per_epoch=3 \
  --model_dir=${DATA_DIR}/ner \
  --distribution_strategy=mirrored \
  --init_checkpoint=${BERT_DIR}/bert_model-1 \
  --num_gpus=1 \
  --train_data_size=10748 \
  --eval_data_size=1343 \
  # --dtype=fp16 \
  # --loss_scale=dynamic \


