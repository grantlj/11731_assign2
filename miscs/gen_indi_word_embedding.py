'''
    Generate the individual word-embeddings.
'''
import os
import sys
sys.path.append("../")
import pickle
import numpy as np
from vocab import Vocab, VocabEntry
import utils
DIM=300

lang2id_fn="../lang/lang2id.pkl"
lang_dict=utils.read_dict_from_pkl(lang2id_fn)

vocab_root_path="../data_ted/vocab/"
embed_root_path="../word_embed/"
assert os.path.exists(vocab_root_path)
assert os.path.exists(embed_root_path)


def handle_a_lang(lang,id):
    pass
    vocab_fn=os.path.join(vocab_root_path,str(id)+".lowercase.vocab.pkl")
    assert os.path.isfile(vocab_fn)
    vocab=utils.read_dict_from_pkl(vocab_fn)

    size = len(vocab)
    dim = DIM
    ret_mat = np.random.rand(size, dim)

    # ret_mat = np.zeros((size, dim))
    not_found_size = size
    embed_fn=os.path.join(embed_root_path,"wiki."+lang+".vec")

    with open(embed_fn, "r") as f:
        line_ind = 0
        while True:
            line_ind += 1
            cur_line = f.readline()
            if cur_line == "" or cur_line is None:
                break
            if line_ind == 1:
                continue

            cur_line = cur_line.replace("\n", "")
            cur_line = cur_line.replace("\r", "")
            tmp_str = cur_line.split(" ")

            word = tmp_str[0];
            embed = tmp_str[1:len(tmp_str) - 1]
            embed = [float(x) for x in embed]
            embed = np.asarray(embed)

            if not word in vocab.word2id:
                # not_exist+=1
                # print("Word: ",word," not existed...")
                continue

            word_indice = vocab.word2id[word]
            ret_mat[word_indice, :] = embed
            not_found_size -= 1
            assert len(embed) == dim

    print("not existed ratio:", float(not_found_size), "/", float(size))
    return ret_mats

if __name__=="__main__":
    for lang,id in lang_dict.items():
        print("Handling language: ",lang)
        lang_mat=handle_a_lang(lang,id)
        dst_fn=os.path.join(embed_root_path,lang+".npz")
        np.savez_compressed(dst_fn,embedding=lang_mat)

    print('done.')