#!/usr/bin/env python
"""
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle
import os
import sys
sys.path.append("../")


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

#   write dict or other objects to file
def write_dict_to_pkl(obj,fn):
    if not fn.endswith(".pkl"):
        fn=fn+".pkl"
    with open(fn,"wb") as f:
        pickle.dump(obj,f)

#   read object from dict
def read_dict_from_pkl(fn):
    if not fn.endswith(".pkl"):
        fn=fn+".pkl"

    with open(fn,"rb") as f:
        obj=pickle.load(f)
    return obj

class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    #   this is another thing we want
    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    #   this is the thing we want.
    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self,sent_ids):
        ret_words=[]
        for ind in sent_ids:
            ret_words.append(self.id2word[ind])
        return ret_words


    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2,lower_case=True):
        vocab_entry = VocabEntry()

        if lower_case:
            for sent_id in range(0,len(corpus)):
                for w_id in range(0,len(corpus[sent_id])):
                    corpus[sent_id][w_id]=corpus[sent_id][w_id].lower()

        #   word freq less than something, just go away
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, src_sents, tgt_sents, vocab_size, freq_cutoff,lower_case):
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        self.src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff,lower_case=lower_case)

        print('initialize target vocabulary ..')
        self.tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff,lower_case=lower_case)

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


'''
#   pointwise
if __name__=="__main__":
    name2id_dict=read_dict_from_pkl("lang/lang2id.pkl")

    vocab_size=50000
    freq_cutoff=2
    lower_case=True
    text_fn="data_ted/train.en-ru.ru.txt"
    lang="ru"
    lang_id=name2id_dict[lang]
    dst_root_path="data_ted/vocab/"

    if lower_case:
        dst_fn = os.path.join(dst_root_path, str(lang_id) + ".lowercase.vocab")
    else:
        dst_fn=os.path.join(dst_root_path,str(lang_id)+".vocab")
    sents=read_corpus(text_fn,"src")

    vocab=VocabEntry.from_corpus(sents,vocab_size,freq_cutoff,lower_case=lower_case)
    write_dict_to_pkl(vocab,dst_fn)
    print("done.")
'''

#   PAIRWISE 
if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    #   source sentences, and the target sentences (they are pairs, english and germany, respectively
    src_sents = read_corpus(args['--train-src'], source='src')
    tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

    #   generate vocabulary for both src and targets （this is a class)
    vocab = Vocab(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']),lower_case=True)
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    pickle.dump(vocab, open(args['VOCAB_FILE'], 'wb'))
    print('vocabulary saved to %s' % args['VOCAB_FILE'])

