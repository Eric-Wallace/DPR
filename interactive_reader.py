#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import sys
sys.path.append('../.')
sys.path.append('./')
from interactive_openbook import get_docs
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
import collections
import glob
import json
import logging
import os
from collections import defaultdict
from typing import List
import time

import numpy as np
import torch
import torch.nn.functional as F

from dpr.data.qa_validation import exact_match_score
from dpr.data.reader_data import (
    ReaderSample,
    ReaderPassage,
    get_best_spans,
    SpanPrediction,
    convert_retriever_results,
)
from dpr.models import init_reader_components
from dpr.models.reader import create_reader_input, ReaderBatch, compute_loss
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    set_seed,
    add_training_params,
    add_reader_preprocessing_params,
    set_encoder_params_from_state,
    get_encoder_params_state,
    add_tokenizer_params,
    print_args,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
    read_serialized_data_from_files,
    Tensorizer,
)
from dpr.utils.model_utils import (
    get_schedule_linear,
    load_states_from_checkpoint,
    move_to_device,
    CheckpointState,
    get_model_file,
    setup_for_distributed_mode,
    get_model_obj,
)

ReaderQuestionPredictions = collections.namedtuple("ReaderQuestionPredictions", ["id", "predictions", "gold_answers"])

def _pad_to_len(seq, pad_id, max_len):
    s_len = seq.size(0)
    if s_len > max_len:
        return seq[0: max_len]
    return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)

def _concat_pair(t1, t2, middle_sep=None, tailing_sep=None):
    middle = [middle_sep] if middle_sep else []
    r = [t1] + middle + [t2] + ([tailing_sep] if tailing_sep else [])
    return torch.cat(r, dim=0), t1.size(0) + len(middle)

class Reader(object):
    def __init__(self, args, model_file):
        self.args = args

        saved_state = load_states_from_checkpoint(model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, reader, optimizer = init_reader_components(args.encoder_model_type, args)
        tensorizer.pad_to_max = False
        del optimizer

        reader = reader.cuda()
        reader = reader.eval()
        self.reader = reader
        self.tensorizer = tensorizer

        model_to_load = get_model_obj(self.reader)
        model_to_load.load_state_dict(saved_state.model_dict)

    def _get_best_prediction(
        self,
        start_logits,
        end_logits,
        relevance_logits,
        samples_batch: List[ReaderSample],
        passage_thresholds: List[int] = None,
    ) -> List[ReaderQuestionPredictions]:

        args = self.args
        max_answer_length = 10
        # max_answer_length = args.max_answer_length
        questions_num, passages_per_question = relevance_logits.size()

        _, idxs = torch.sort(
            relevance_logits,
            dim=1,
            descending=True,
        )

        batch_results = []
        for q in range(questions_num):
            sample = samples_batch[q]

            non_empty_passages_num = len(sample.passages)
            nbest = []
            for p in range(passages_per_question):
                passage_idx = idxs[q, p].item()
                if (
                    passage_idx >= non_empty_passages_num
                ):  # empty passage selected, skip
                    continue
                reader_passage = sample.passages[passage_idx]
                sequence_ids = reader_passage.sequence_ids
                sequence_len = sequence_ids.size(0)
                # assuming question & title information is at the beginning of the sequence
                passage_offset = reader_passage.passage_offset

                p_start_logits = start_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]
                p_end_logits = end_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]

                ctx_ids = sequence_ids.tolist()[passage_offset:]
                best_spans = get_best_spans(
                    self.tensorizer,
                    p_start_logits,
                    p_end_logits,
                    ctx_ids,
                    max_answer_length,
                    passage_idx,
                    relevance_logits[q, passage_idx].item(),
                    top_spans=10,
                )
                nbest.extend(best_spans)
                if False and len(nbest) > 0 and not passage_thresholds:
                    break

            #if passage_thresholds:
            #    passage_rank_matches = {}
            #    for n in passage_thresholds:

            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO: Consider changing scoring function to softmax earlier (incl. over multiple paragraphs)
            # TODO:
            # TODO:
            # TODO:
            # TODO: casing
            
            # Softmax all scores
            scores = []
            for pred in nbest:
                scores.append(pred.span_score)
            smax_scores = F.softmax(torch.Tensor(scores)).tolist()
            for i in range(len(nbest)):
                pred = nbest[i]
                nbest[i] = pred._replace(span_score = smax_scores.pop(0))
            curr_nbest_dict = {}

            # Add duplicates
            for pred in nbest:
                if pred.prediction_text in curr_nbest_dict.keys():
                    curr_nbest_dict[pred.prediction_text] = curr_nbest_dict[pred.prediction_text]._replace(span_score = pred.span_score + curr_nbest_dict[pred.prediction_text].span_score) # Convoluted thing to just add the two span scores
                else:
                    curr_nbest_dict[pred.prediction_text] = pred

            curr_nbest = sorted(curr_nbest_dict.values(), key=lambda x: -x.span_score)
            #        passage_rank_matches[n] = curr_nbest[0]
            #    predictions = passage_rank_matches
            #else:
            #    if len(nbest) == 0:
            #        predictions = {
            #            passages_per_question: SpanPrediction("", -1, -1, -1, "")
            #        }
            #    else:
            #        predictions = {passages_per_question: nbest[0]}
            predictions = {passages_per_question: curr_nbest}
            batch_results.append(
                ReaderQuestionPredictions(sample.question, predictions, sample.answers)
            )
        return batch_results


def answer_clue(clue, max_answers):
    # TODO, add with no_grad() everywhere
    #
    #
    t0 = time.perf_counter()
    documents = get_docs(clue, max_docs=int(max_answers / 10))
    t1 = time.perf_counter()
    print(f"get_docs took {t1 -t0} seconds")

    passage_ids = []
    for i, mydocument in enumerate(documents):
        sample = ReaderPassage(id=i, text=mydocument[0], title=mydocument[1])
        question_and_title = reader.tensorizer.text_to_tensor(sample.title, title=clue, add_special_tokens=True)
        sample.passage_token_ids = reader.tensorizer.text_to_tensor(sample.passage_text, add_special_tokens=False)
        use_tailing_sep = False
        all_concatenated, shift = _concat_pair(question_and_title, sample.passage_token_ids, tailing_sep=retriever.tensorizer.get_pair_separator_ids() if use_tailing_sep else None)
        sample.sequence_ids = _pad_to_len(all_concatenated, reader.tensorizer.get_pad_id(), 350)
        sample.passage_offset = shift
        assert shift > 1
        passage_ids.append(sample)
    input_ids = torch.stack([t.sequence_ids for t in passage_ids], dim=0)
    input_ids = input_ids.unsqueeze(0).cuda()
    #samples_batch = ReaderBatch(input_ids, None, None, None)
    #input = ReaderBatch(**move_to_device(samples_batch._asdict(), torch.device('cuda')))
    attn_mask = reader.tensorizer.get_attn_mask(input_ids).cuda()
    start_logits, end_logits, relevance_logits = reader.reader(input_ids, attn_mask)
    reader_sample = ReaderSample(clue, None, None, None, passage_ids)
    batch_predictions = reader._get_best_prediction(start_logits, end_logits, relevance_logits, [reader_sample])
    answers = [span.prediction_text for span in list(batch_predictions[0].predictions.values())[0]][0:max_answers]
    return answers

reader = None
def setup_reader(model_file):
    global reader
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)

    args = parser.parse_args()

    setup_args_gpu(args)
    set_seed(args)
    print_args(args)
    reader = Reader(args, model_file)

