import math
from typing import List

import numpy as np
import copy

import json
import pickle

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


def save_model_by_state_dict(model,model_fn):
    pass
    import torch
    torch.save({'state_dict':model.cpu().state_dict()},model_fn)
    model=model.cuda()
    return

#   load model state dict
def load_model_by_state_dict(model,state_dict_fn):
    pass
    import torch
    model_dict=torch.load(state_dict_fn)
    model.load_state_dict(model_dict['state_dict'])
    return model


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t

def input_transpose_max_len(sents, pad_token,MAX_LEN):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch

    We abandon all those sentenses with length>max_len
    """

    batch_size = len(sents)

    sents_t = []
    for i in range(MAX_LEN):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def get_type_from_fn(fp):
    tmp_str=fp.split(".")
    return tmp_str[-2]

def read_corpus_multi_src(file_path, source):
    name2id_dict=read_dict_from_pkl("lang/lang2id.pkl")
    type_by_src=get_type_from_fn(file_path)
    id_by_src=None
    type_list=None
    try:
        #   a single language
        id_by_src=name2id_dict[type_by_src]
    except:
        #   multi language, need to parse the type pkl further
        type_fn=file_path[0:-8]+"type.pkl"
        type_list=read_dict_from_pkl(type_fn)


    data = []
    for lid,line in enumerate(open(file_path)):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']

        if not id_by_src is None:
            data.append((id_by_src,sent))
            continue

        if source=="tgt":
            cur_id=name2id_dict[type_list[lid]['tgt']]
        else:
            cur_id=name2id_dict[type_list[lid]['src']]

        data.append((cur_id,sent))

    return data



#  LJ: A method to yield the training batches (NOTE: the shuffle is False here)
def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

#  LJ: A method to yield the training batches (NOTE: the shuffle is False here)
def batch_iter_multi_src(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        src_sents = [e[0][1] for e in examples]
        tgt_sents = [e[1][1] for e in examples]
        src_ids = [e[0][0] for e in examples]
        tgt_ids = [e[1][0] for e in examples]

        yield src_sents, tgt_sents, src_ids, tgt_ids

if __name__=="__main__":
    train_data=read_corpus_multi_src("/home/jiangl1/11731_assign2/data_ted/test.en-glpt.glpt.txt",source="src")
    for src_sents,tgt_sents,src_ids,tgt_ids in batch_iter_multi_src(train_data,batch_size=30,shuffle=True):
        print(src_sents,tgt_sents,src_ids,tgt_ids)
    print("done.")