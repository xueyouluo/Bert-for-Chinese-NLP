# 将models的路径加入到PYTHONPATH环境变量中
export PYTHONPATH="$PYTHONPATH:/nfs/users/xueyou/github/models"
export BERT_V1_PATH=/data/xueyou/data/bert_pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12
export BERT_V2_PATH=/data/xueyou/data/bert_pretrain/converted/chinese_roberta_wwm_ext_L-12_H-768_A-12

# convert
python scripts/tf2_encoder_checkpoint_converter.py \
  --bert_config_file=${BERT_V1_PATH}/bert_config.json \
  --checkpoint_to_convert=${BERT_V1_PATH}/bert_model.ckpt \
  --converted_checkpoint_path=${BERT_V2_PATH}/bert_model

# copy assets
cp ${BERT_V1_PATH}/bert_config.json ${BERT_V2_PATH}
cp ${BERT_V1_PATH}/vocab.txt ${BERT_V2_PATH}