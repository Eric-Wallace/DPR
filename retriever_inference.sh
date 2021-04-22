# Running retriever inference to build training set for another retriever

FILES=retriever_input/*

for f in ${FILES}; do
	CUDA_VISIBLE_DEVICES=9 python dense_retriever.py --model_file data/embeddings/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp --qa_file ${f} --ctx_file data/wikipedia_split/psgs_w100.tsv --encoded_ctx_file "data/embeddings/downloads/data/retriever_results/nq/single-adv-hn/wikipedia_passages_*.pkl" --out_file retriever_output/$(basename ${f}) --save_or_load_index --batch_size 1000 --n-docs 25
done


# CUDA_VISIBLE_DEVICES=9 python dense_retriever.py --model_file checkpoint/retriever/dpr_biencoder.11.87 --qa_file small.tsv --ctx_file data/wikipedia_split/psgs_w100.tsv --encoded_ctx_file "data/embeddings/downloads/data/retriever_results/nq/single-adv-hn/wikipedia_passages_*.pkl" --out_file fuck.json --save_or_load_index --batch_size 1000 --n-docs 20
