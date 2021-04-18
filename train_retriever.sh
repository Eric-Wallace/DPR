TRAIN_FILE="data/retriever_train.json"
VALID_FILE="data/retriever_valid.json" # also could be folder
OUTPUT_DIR="checkpoint/retriever/"
PRETRAINED_MODEL="data/embeddings/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp"
LOG_FILE="logs/train-$(date +%s).txt" 

#	--model_file ${PRETRAINED_MODEL} \
# TODO: Add extra featuers for fill len + stuff, then increase seqlen
CUDA_VISIBLE_DEVICES=1 python3 train_dense_encoder.py \
	--batch_size 128 \
	--dev_batch_size 128 \
	--sequence_length 256 \
	--max_grad_norm 2.0 \
	--warmup_steps 1237 \
	--learning_rate 2e-05 \
	--seed 1 \
	--global_loss_buf_sz 1000000 \
	--num_train_epochs 60 \
	--do_lower_case \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--fix_ctx_encoder \
	--train_file ${TRAIN_FILE} \
	--dev_file ${VALID_FILE} \
	--output_dir ${OUTPUT_DIR}  | tee ${LOG_FILE}
