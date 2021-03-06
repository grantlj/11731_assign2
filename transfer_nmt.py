# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    transfer_nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    transfer_nmt.py decode --vocab=<file> [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    transfer_nmt.py decode --vocab=<file> [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --src_lang=<string>                     the list of source languages, e.g. "en,tr,ru"
    --tgt_lang=<string>                     the target language type [default: en]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 3]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --src_ebed_fn=<file>                    LJ: source word embedding [default: None]
    --tgt_ebed_fn=<file>                    LJ: target word embedding [default: None]

"""
import sys
sys.path.append("../")
import math
import pickle
import sys
import time
from collections import namedtuple
from typing import Any

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import utils
from utils import read_corpus, batch_iter,batch_iter_multi_src,read_corpus_multi_src
from vocab import Vocab, VocabEntry
import torch
import torch.nn.functional as F
import pdb
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils import clip_grad_norm

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

MAX_LEN = 100


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, loss,
                 src_id_list,tgt_id,
                 dropout_rate=0.2):
        super(NMT, self).__init__()

        #   LJ: the source language ids
        self.low_src_id = src_id_list[0]
        self.high_src_id = src_id_list[1]

        #   LJ: the target language id
        self.tgt_id=tgt_id

        #self.bi_direct = bi_direct
        #self.conv = conv
        #self.local_att = local_att
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.tgt_vocab = vocab[2]
        self.low_src_vocab = vocab[0]
        self.high_src_vocab = vocab[1]
        #self.decoding_type = decoding_type
        self.loss = loss
        self.tgt_vocab_size = len(vocab[2].word2id)
        self.low_src_embed = nn.Embedding(len(vocab[0].word2id), embed_size, padding_idx=0).cuda()
        self.high_src_embed = nn.Embedding(len(vocab[1].word2id), embed_size, padding_idx=0).cuda()
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, embed_size, padding_idx=0).cuda()
        self.word_dist = nn.Linear(embed_size, self.tgt_vocab_size).cuda()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.high_transform = nn.Linear(embed_size, embed_size).cuda()

        self.cpu_time = 0

        self.key_size = 50

        self.low_encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True).cuda()
        self.high_encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True).cuda()
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True).cuda()
        self.decoder = nn.LSTM(embed_size * 2, hidden_size).cuda()
        self.a_key = nn.Linear(hidden_size, self.key_size * 2).cuda()
        self.out = nn.Linear(hidden_size + embed_size, embed_size).cuda()

        self.q_key = nn.Linear(hidden_size * 2, self.key_size).cuda()
        self.q_value = nn.Linear(hidden_size * 2, embed_size).cuda()
        self.q_low_key = nn.Linear(hidden_size * 2, self.key_size).cuda()
        self.q_low_value = nn.Linear(hidden_size * 2, embed_size).cuda()
        self.q_high_key = nn.Linear(hidden_size * 2, self.key_size).cuda()
        self.q_high_value = nn.Linear(hidden_size * 2, embed_size).cuda()

        self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size).cuda()
        self.cell_transform = nn.Linear(hidden_size * 2, hidden_size).cuda()
        self.low_hidden_transform = nn.Linear(hidden_size * 2, hidden_size).cuda()
        self.low_cell_transform = nn.Linear(hidden_size * 2, hidden_size).cuda()
        self.high_hidden_transform = nn.Linear(hidden_size * 2, hidden_size).cuda()
        self.high_cell_transform = nn.Linear(hidden_size * 2, hidden_size).cuda()

    def forward(self, src_sents: List[List[str]], lang_label: List[int], tgt_sents: List[List[str]], keep_grad=True) -> Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        pairs = list(zip(src_sents, tgt_sents, lang_label))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        src_sents, tgt_sents, lang_label = zip(*pairs)
        lang_label = np.asarray(lang_label, dtype=np.int32)
        src_lengths = np.asarray([len(s) for s in src_sents], dtype=np.int32)
        tgt_lengths = np.asarray([len(s) for s in tgt_sents], dtype=np.int32)
        src_max_len = max(src_lengths)
        tgt_max_len = max(tgt_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, src_max_len).long()
        tgt_ind = torch.zeros(batch_size, tgt_max_len).long()
        for x in range(len(src_sents)):
            if lang_label[x] == self.low_src_id:
                src_ind[x, :len(src_sents[x])] = torch.LongTensor(self.low_src_vocab.words2indices(src_sents[x]))
            elif lang_label[x] == self.high_src_id:
                src_ind[x, :len(src_sents[x])] = torch.LongTensor(self.high_src_vocab.words2indices(src_sents[x]))
            else:
                raise NotImplementedError
            tgt_ind[x, :len(tgt_sents[x])] = torch.LongTensor(self.tgt_vocab.words2indices(tgt_sents[x]))
        src_ind = src_ind.cuda()
        tgt_ind = tgt_ind.cuda()

        def train():
            low_ind = np.where(lang_label == self.low_src_id)[0]
            high_ind = np.where(lang_label == self.high_src_id)[0]

            low_src_encodings, low_decoder_init_state, low_src_embed = self.encode(src_ind[low_ind], src_lengths[low_ind], 'low_')
            src_embed = torch.zeros(batch_size, src_max_len, self.embed_size).cuda()
            src_embed[low_ind] = low_src_embed
            src_spec_encodings = torch.zeros(batch_size, src_max_len, self.hidden_size * 2).cuda()
            src_spec_encodings[low_ind, :low_src_encodings.size(1), :] = low_src_encodings
            decoder_spec_init_cell = torch.zeros(2, batch_size, self.hidden_size).cuda()
            decoder_spec_init_hidden = torch.zeros(2, batch_size, self.hidden_size).cuda()
            decoder_spec_init_cell[:, low_ind, :] = low_decoder_init_state[0]
            decoder_spec_init_hidden[:, low_ind, :] = low_decoder_init_state[1]
            if len(high_ind):
                high_src_encodings, high_decoder_init_state, high_src_embed = self.encode(src_ind[high_ind], src_lengths[high_ind], 'high_')
                src_embed[high_ind] = high_src_embed
                src_spec_encodings[high_ind, :high_src_encodings.size(1), :] = high_src_encodings
                decoder_spec_init_cell[:, high_ind, :] = high_decoder_init_state[0]
                decoder_spec_init_hidden[:, high_ind, :] = high_decoder_init_state[1]
            decoder_spec_init_state = (decoder_spec_init_cell, decoder_spec_init_hidden)

            src_encodings, decoder_init_state, src_embed = self.encode(src_ind, src_lengths, '', src_embed=src_embed)
            loss, num_words = self.decode((src_spec_encodings, src_encodings), src_lengths, (decoder_spec_init_state, decoder_init_state), tgt_ind, tgt_lengths, low_ind, high_ind)
            return loss, num_words

        if keep_grad:
            #   for training stage
            loss, num_words = train()
        else:
            #   for test stage
            with torch.no_grad():
                loss, num_words = train()

        return loss, num_words

    def init_hidden(self, hidden, batch_size, low_ind, high_ind):
        h = hidden[1][0].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2).unsqueeze(0)
        c = hidden[1][1].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2).unsqueeze(0)
        h_t = self.hidden_transform(h)
        c_t = self.cell_transform(h)

        spec_h = hidden[0][0].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2).unsqueeze(0)
        spec_c = hidden[0][1].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2).unsqueeze(0)
        spec_h_t = torch.zeros(1, batch_size, self.hidden_size).cuda()
        spec_c_t = torch.zeros(1, batch_size, self.hidden_size).cuda()
        spec_h_t[:, low_ind, :] = self.low_hidden_transform(spec_h[:, low_ind, :])
        spec_c_t[:, low_ind, :] = self.low_cell_transform(spec_c[:, low_ind, :])
        if len(high_ind):
            spec_h_t[:, high_ind, :] = self.high_hidden_transform(spec_h[:, high_ind, :])
            spec_c_t[:, high_ind, :] = self.high_cell_transform(spec_c[:, high_ind, :])
        return (F.tanh(h_t + spec_h_t), F.tanh(c_t + spec_c_t))

    def encode(self, src_sents: List[List[str]], src_lengths, name, keep_grad=True, src_embed=None) -> Tuple[Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in other formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        if src_embed is None:
            src_embed = eval('self.{}src_embed(src_sents)'.format(name))
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = eval('self.{}encoder(packed_input)'.format(name))
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)

        return src_hidden, src_last_hidden, src_embed

    def attention(self, decoder_hidden, q_gen_key, q_gen_value, q_spec_key, q_spec_value, q_mask):
        a_key = self.a_key(decoder_hidden[0].squeeze(0))

        q_key = torch.cat((q_gen_key, q_spec_key), dim=2)
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_gen_value + q_spec_value)

        return q_context

    def decode(self, src_encodings: Tensor, src_lengths, decoder_init_state: Any, tgt_sents: List[List[str]],
               tgt_lengths, low_ind, high_ind, keep_grad=True):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        '''
        src_encodings = torch.cat(src_encodings, dim=2)
        decoder_init_state = (torch.cat((decoder_init_state[0][0], decoder_init_state[1][0]), dim=2),
                              torch.cat((decoder_init_state[0][1], decoder_init_state[1][1]), dim=2))
        '''

        batch_size = src_encodings[0].size(0)
        length = src_encodings[0].size(1)
        tgt_embed = self.tgt_embed(tgt_sents)
        tgt_l = tgt_embed.size(1)

        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = torch.cuda.FloatTensor(batch_size, tgt_l - 1, self.hidden_size + self.embed_size)
        decoder_hidden = self.init_hidden(decoder_init_state, batch_size, low_ind, high_ind)

        q_key = self.q_key(src_encodings[1])
        q_value = self.q_value(src_encodings[1])
        q_spec_key = torch.zeros(batch_size, length, self.key_size).cuda()
        q_spec_key[low_ind, :, :] = self.q_low_key(src_encodings[0][low_ind, :, :])
        q_spec_value = torch.zeros(batch_size, length, self.embed_size).cuda()
        q_spec_value[low_ind, :, :] = self.q_low_value(src_encodings[0][low_ind, :, :])
        if len(high_ind):
            q_spec_key[high_ind, :, :] = self.q_high_key(src_encodings[0][high_ind, :, :])
            q_spec_value[high_ind, :, :] = self.q_high_value(src_encodings[0][high_ind, :, :])
        q_mask = torch.arange(length).long().cuda().repeat(batch_size, 1) < torch.cuda.LongTensor(
            src_lengths).repeat(length, 1).transpose(0, 1)

        src_lengths = torch.cuda.LongTensor(src_lengths)
        for step in range(tgt_l - 1):
            context = self.attention(decoder_hidden, q_key, q_value, q_spec_key, q_spec_value, q_mask)
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1),
                                                          decoder_hidden)

            decoder_outputs[:, step, :] = torch.cat((decoder_output.transpose(0, 1), context), dim=2).squeeze(1)
            decoder_input = tgt_embed[:, step + 1, :].unsqueeze(1)

        logits = self.word_dist(F.tanh(self.out(self.dropout(decoder_outputs))))
        logits = F.log_softmax(logits, dim=2)
        logits = logits.contiguous().view(-1, self.tgt_vocab_size)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        return loss, (tgt_sents[:, 1:] != 0).sum().item()

    def beam_search(self, src_sent: List[str], beam_size: int = 20, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        self.eou = 2
        top_k = 20
        batch_size = 1
        low_ind = [0]
        high_ind = []

        src_ind = torch.cuda.LongTensor(self.low_src_vocab.words2indices(src_sent[1]))
        src_embed = self.low_src_embed(src_ind).unsqueeze(0)
        
        src_lengths = np.asarray([len(src_sent[1])])
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        low_src_output, low_src_last_hidden = self.low_encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        low_src_hidden, _ = pad_packed_sequence(low_src_output, batch_first=True)
        decoder_hidden = self.init_hidden((low_src_last_hidden, src_last_hidden), batch_size, low_ind, high_ind)

        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.tgt_embed(torch.cuda.LongTensor([1])).unsqueeze(1)
        length = src_hidden.size(1)
        src_lengths = torch.cuda.LongTensor(src_lengths)

        q_key = self.q_key(src_hidden)
        q_value = self.q_value(src_hidden)
        q_spec_key = torch.zeros(batch_size, length, self.key_size).cuda()
        q_spec_key[low_ind, :, :] = self.q_low_key(low_src_hidden[low_ind, :, :])
        q_spec_value = torch.zeros(batch_size, length, self.embed_size).cuda()
        q_spec_value[low_ind, :, :] = self.q_low_value(low_src_hidden[low_ind, :, :])
        q_mask = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(
            src_lengths).repeat(length, 1).transpose(0, 1)
        
        context = self.attention(decoder_hidden, q_key, q_value, q_spec_key, q_spec_value, q_mask)
        decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output = torch.cat((decoder_output, context), dim=2)
        decoder_output = self.word_dist(F.tanh(self.out(decoder_output.squeeze(1))))
        decoder_output[:, 0] = -np.inf

        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        beam = torch.zeros(beam_size, max_decoding_time_step).long().cuda()
        beam[:, 0] = argtop
        beam_probs = logprobs.clone().squeeze(0)
        beam_eos = argtop.squeeze(0) == self.eou
        decoder_hidden = (decoder_hidden[0].expand(1, beam_size, self.hidden_size).contiguous(),
                          decoder_hidden[1].expand(1, beam_size, self.hidden_size).contiguous())
        decoder_input = self.tgt_embed(argtop.squeeze(0)).unsqueeze(1)

        src_hidden = src_hidden.expand(beam_size, length, self.hidden_size * 2)
        low_src_hidden = low_src_hidden.expand(beam_size, length, self.hidden_size * 2)
        q_key = self.q_key(src_hidden)
        q_value = self.q_value(src_hidden)
        q_spec_key = self.q_low_key(low_src_hidden)
        q_spec_value = self.q_low_value(low_src_hidden)
        q_mask = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(
            src_lengths).repeat(length, 1).transpose(0, 1)

        for t in range(max_decoding_time_step - 1):
            context = self.attention(decoder_hidden, q_key, q_value, q_spec_key, q_spec_value, q_mask)
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1),
                                                          decoder_hidden)
            decoder_output = torch.cat((decoder_output.transpose(0, 1), context), dim=2)
            decoder_output = self.word_dist(F.tanh(self.out(decoder_output)))

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.expand(top_k, beam_size).transpose(0, 1) + logprobs).view(-1).topk(
                beam_size)

            last = best_args / top_k
            curr = best_args % top_k
            beam[:, :] = beam[last, :]
            beam_eos = beam_eos[last]
            beam_probs = beam_probs[last]
            beam[:, t + 1] = argtop[last, curr] * (~beam_eos).long() + eos_filler * beam_eos.long()
            mask = ~beam_eos
            beam_probs[mask] = (beam_probs[mask] * (t + 1) + best_probs[mask]) / (t + 2)
            decoder_hidden = (decoder_hidden[0][:, last, :], decoder_hidden[1][:, last, :])

            beam_eos = beam_eos | (beam[:, t + 1] == self.eou)
            decoder_input = self.tgt_embed(beam[:, t + 1]).unsqueeze(1)

            if beam_eos.all():
                break

        best, best_arg = beam_probs.max(0)
        translation = beam[best_arg].cpu().tolist()
        if self.eou in translation:
            translation = translation[:translation.index(self.eou)]
        translation = [self.tgt_vocab.id2word[w] for w in translation]
        return [Hypothesis(value=translation, score=best.item())]

    def evaluate_ppl(self, dev_data: List[Any], batch_size: int = 32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        self.set_model_to_eval()

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents,src_ids,tgt_ids in batch_iter_multi_src(dev_data, batch_size):
            loss, num_words = self.__call__(src_sents, src_ids, tgt_sents, keep_grad=False)

            loss = loss.detach().cpu().numpy()

            cum_loss += loss * num_words
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return cum_loss, ppl

    #   set model to train and test state
    def set_model_to_train(self):
        self.train()
        return

    #   set model to validation
    def set_model_to_eval(self):
        self.eval()
        return

    def load(self, model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        encoder_fn = model_path.replace(".bin", "_encoder.pkl")
        decoder_fn = model_path.replace(".bin", "_decoder.pkl")

        self.encoder = utils.load_model_by_state_dict(self.encoder, encoder_fn)
        self.decoder = utils.load_model_by_state_dict(self.decoder, decoder_fn)

        self.encoder.cuda().eval()
        self.decoder.cuda().eval()

        return

    def save(self, path: str):
        """
        Save current model to file
        """
        '''
        encoder_fn=path.replace(".bin","_encoder.pkl")
        decoder_fn=path.replace(".bin","_decoder.pkl")

        utils.save_model_by_state_dict(self.encoder,encoder_fn)
        utils.save_model_by_state_dict(self.decoder,decoder_fn)
        '''
        utils.save_model_by_state_dict(self, path)

        return


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


#   LJ: the training process starting here.
def train(args: Dict[str, str]):

    #   LJ: source corpus and target corpus
    train_data_src = read_corpus_multi_src(args['--train-src'], source='src')
    train_data_tgt = read_corpus_multi_src(args['--train-tgt'], source='tgt')

    #   LJ: the validation set (source and target)
    dev_data_src = read_corpus_multi_src(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus_multi_src(args['--dev-tgt'], source='tgt')

    #   LJ: the training and validation sentences pairs
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    #   LJ: the configurations
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    #   LJ: read the vocabulary
    #vocab = pickle.load(open(args['--vocab'], 'rb'))

    #   LJ: set up the loss function (ignore to <pad>)
    nll_loss = nn.NLLLoss(ignore_index=0)

    #   LJ: add support for multiple src inputs
    src_name_list=args['--src_lang'].split(",")
    tgt_name=args['--tgt_lang']

    #   read the name2id mapping
    lang2id_dict=utils.read_dict_from_pkl("./lang/lang2id.pkl")
    src_id_list=[lang2id_dict[x] for x in src_name_list]
    tgt_id=lang2id_dict[tgt_name]
    vocab = (pickle.load(open('./data_ted/vocab/{}.vocab'.format(src_id_list[0]), 'rb')),
             pickle.load(open('./data_ted/vocab/{}.vocab'.format(src_id_list[1]), 'rb')),
             pickle.load(open('./data_ted/vocab/{}.vocab'.format(tgt_id), 'rb')))

    #   LJ: build the model
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                src_id_list=src_id_list,
                tgt_id=tgt_id,
                vocab=vocab,
                loss=nll_loss)


    bound = float(args['--uniform-init'])
    for p in model.parameters():
        torch.nn.init.uniform_(p.data, a=-bound, b=bound)

    src_embed_fn=args['--src_ebed_fn']
    tgt_embed_fn=args['--tgt_ebed_fn']

    #print(src_embed_fn)
    #print(tgt_embed_fn)

    if not src_embed_fn=="None":
        src_vectors = np.load(src_embed_fn)['embedding']
        model.src_embed.weight.data = torch.from_numpy(src_vectors).float().cuda()

    if not tgt_embed_fn=="None":
        tgt_vectors = np.load(tgt_embed_fn)['embedding']
        model.tgt_embed.weight.data = torch.from_numpy(tgt_vectors).float().cuda()

    #   LJ: the learning rate
    lr = float(args['--lr'])

    #   LJ: setting some initial losses, etc.
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    #   LJ: setup the optimizer
    # optimizer = optim.Adam(list(model.encoder.parameters())+list(model.decoder.parameters()), lr=lr)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)

    while True:

        #   start the epoch
        epoch += 1

        #   LJ: ok, we yield the sentences in a shuffle manner.
        for src_sents, tgt_sents,src_ids,tgt_ids in batch_iter_multi_src(train_data, batch_size=train_batch_size, shuffle=True):

            model.set_model_to_train()

            train_iter += 1

            #   LJ: current batch size
            batch_size = len(src_sents)

            # (batch_size)
            # LJ: train on the mini-batch and get the loss, backpropagation

            # loss = -model(src_sents, tgt_sents)
            optimizer.zero_grad()
            loss, num_words = model(src_sents, src_ids, tgt_sents)
            loss.backward()
            #clip_grad_norm(list(model.encoder.parameters()) + list(model.decoder.parameters()), clip_grad)
            clip_grad_norm(model.parameters(), clip_grad)
            optimizer.step()

            #   add the loss to cumlinative loss
            report_loss += loss.detach().cpu().numpy() * num_words
            cum_loss += loss.detach().cpu().numpy() * num_words

            #   LJ: how many targets words are there in all target sentences in current batch
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`

            #   LJ: all cumulative words
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict

            #   LJ: all number of instances handled
            report_examples += batch_size
            cumulative_examples += batch_size

            #   LJ: print out the training loss
            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(
                                                                                             report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (
                                                                                                     time.time() - train_time),
                                                                                         time.time() - begin_time),
                      file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                model.set_model_to_eval()
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cumulative_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cumulative_tgt_words),
                                                                                             cumulative_examples),
                      file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                #   LJ: the validation is implemented in a seperate function
                cum_loss, dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)  # dev batch size can be a bit larger
                # valid_metric = -dev_ppl
                valid_metric = -cum_loss

                print('validation: iter %d, dev. ppl %f, val cum loss: %f' % (train_iter, dev_ppl, cum_loss),
                      file=sys.stderr)

                #   LJ: a new better model is found.
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # You may also save the optimizer's state, adjust the training weight, since we found there are too
                    #   much iterations without improvements.
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        # model.load(model_save_path)
                        model = utils.load_model_by_state_dict(model, model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[
    List[Hypothesis]]:
    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


#   LJ: the decode step (i.e., the test stage)
def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus_multi_src(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus_multi_src(args['TEST_TARGET_FILE'], source='tgt')
        test_data_tgt = [data[1] for data in test_data_tgt]

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)

    #   LJ: read the vocabulary
    vocab = pickle.load(open(args['--vocab'], 'rb'))

    #   LJ: set up the loss function (ignore to <pad>)
    nll_loss = nn.NLLLoss(ignore_index=0)

    #   LJ: add support for multiple src inputs
    src_name_list = args['--src_lang'].split(",")
    tgt_name = args['--tgt_lang']

    #   read the name2id mapping
    lang2id_dict = utils.read_dict_from_pkl("./lang/lang2id.pkl")
    src_id_list = [lang2id_dict[x] for x in src_name_list]
    tgt_id = lang2id_dict[tgt_name]
    vocab = (pickle.load(open('./data_ted/vocab/{}.vocab'.format(src_id_list[0]), 'rb')),
             pickle.load(open('./data_ted/vocab/{}.vocab'.format(src_id_list[1]), 'rb')),
             pickle.load(open('./data_ted/vocab/{}.vocab'.format(tgt_id), 'rb')))



    #   LJ: build the model
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab,
                loss=nll_loss,
                src_id_list=src_id_list,
                tgt_id=tgt_id)

    '''
    model.load(args["MODEL_PATH"])
    '''
    model = utils.load_model_by_state_dict(model, args['MODEL_PATH'])

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)
    print(args)
    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
