#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import hashlib
import random
import string
import sys
sys.path.append('../.')
sys.path.append('./')
from dpr.utils.model_utils import move_to_device
from dpr.data.reader_data import (
    ReaderSample,
    ReaderPassage,
    get_best_spans,
    SpanPrediction,
    convert_retriever_results,
)
from dpr.models.reader import ReaderBatch

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)
                    for q in questions[batch_start : batch_start + bsz]
                ]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info("Encoded queries %d", len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        # logger.info("Total encoded queries tensor %s", query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        # logger.info("index search time: %f sec.", time.time() - time0)
        return results

def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info("Reading data from: %s", ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    return docs

def get_docs(clue, max_docs):
    # return [('Nucleic acids are the biopolymers, or small biomolecules, essential to all known forms of life. The term \"nucleic acid\" is the overall name for DNA and RNA. They are composed of nucleotides, which are the monomers made of three components: a 5-carbon sugar, a phosphate group and a nitrogenous base. If the sugar is a compound ribose, the polymer is RNA (ribonucleic acid); if the sugar is derived from ribose as deoxyribose, the polymer is DNA (deoxyribonucleic acid). Nucleic acids are the most important of all biomolecules. They are found in abundance in all living things, where they', 'Nucleic acid')] * max_docs

    #if clue + str(max_docs) in answer_cache:
    #    return answer_cache[clue + str(max_docs)]

    questions = [clue]
    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), max_docs)

    paragraphs = []
    for i in range(len(top_ids_and_scores[0][0])):
        id_ = top_ids_and_scores[0][0][i]
        id_ = id_.replace('wiki:','')
        #score = top_ids_and_scores[0][1][i]
        mydocument = all_passages[id_]
        paragraphs.append(mydocument)
        
        
        #sample = ReaderPassage(id=i, text=mydocument[0], title=mydocument[1]) 
        #sample.passage_token_ids = reader.tensorizer.text_to_tensor(sample.passage_text, add_special_tokens=False)
        #question_and_title = reader.tensorizer.text_to_tensor(sample.title, title=clue, add_special_tokens=True)
        #use_tailing_sep = False
        #all_concatenated, shift = _concat_pair(question_and_title, sample.passage_token_ids, tailing_sep=retriever.tensorizer.get_pair_separator_ids() if use_tailing_sep else None)
        #         positive_input_ids = _pad_to_len(positives[positive_idx].sequence_ids, pad_token_id, max_len)
        #sample.sequence_ids = all_concatenated
        #sample.passage_offset = shift
        #assert shift > 1
        #passages.append(sample)
        ##paragraph = mydocument[0]
        ##title = mydocument[1]
    #input_ids = torch.stack([t for t in passages], dim=0)
    #input_ids = input_ids.unsqueeze(0)
    #ReaderBatch(input_ids, start_positions, end_positions, answers_masks)
    #input = ReaderBatch(**move_to_device(input._asdict(), args.device))
    #attn_mask = retriever.tensorizer.get_attn_mask(input_ids)
    #start_logits, end_logits, relevance_logits = self.reader(input_ids, attn_mask)
    #batch_predictions = self._get_best_prediction(start_logits, end_logits, relevance_logits, samples_batch)
    #answers = [span.prediction_text for span in batch_predictions.predictions.values()]

    #answer_cache[clue + str(max_docs)] = paragraphs
    #if random.random() > 0.97:
    #    with open(answer_cache_path, 'wb') as f:
    #        pickle.dump(answer_cache, f)

    return paragraphs


retriever = None
all_passages = None
answer_cache = None
answer_cache_path = None
def setup_dpr(model_file, ctx_file, encoded_ctx_file, hnsw_index=False, save_or_load_index=False):
    global retriever
    global all_passages
    global answer_cache
    global answer_cache_path
    parameter_setting = model_file + ctx_file + encoded_ctx_file
    answer_cache_path = hashlib.sha1(parameter_setting.encode("utf-8")).hexdigest()
    if os.path.exists(answer_cache_path):
        answer_cache = pickle.load(open(answer_cache_path, 'rb'))
    else:
        answer_cache = {}
    parser = argparse.ArgumentParser()
    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    args = parser.parse_args()
    args.model_file = model_file
    args.ctx_file = ctx_file
    args.encoded_ctx_file = encoded_ctx_file
    args.hnsw_index = hnsw_index
    args.save_or_load_index = save_or_load_index
    args.batch_size = 1 # TODO

    setup_args_gpu(args)
    print_args(args)

    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")

    prefix_len = len("question_model.")
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("question_model.")
    }
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, 50000)
    else:
        index = DenseFlatIndexer(vector_size, 50000, "IVF65536,PQ64") #IVF65536

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and (
        os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")
    ):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index.index_data(input_paths)

        if args.save_or_load_index:
            retriever.index.serialize(index_path)
        # get questions & answers
    
    all_passages = load_passages(args.ctx_file)

