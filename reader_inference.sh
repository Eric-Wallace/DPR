python train_reader.py --seed 42 \
	 --do_lower_case --eval_top_docs 25 \
	--encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased --dev_file "smirnoff.json" --warmup_steps 0 --sequence_length 250 --batch_size 1 --passages_per_question 24 \
	--num_train_epochs 20 --dev_batch_size 72 --passages_per_question_predict 25 --output_dir test --model_file data/hf_bert_base.cp --prediction_results_file "./prediction_results.json"
