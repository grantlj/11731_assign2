import os
import sys
sys.path.append("../")
import utils
from vocab import *
print(utils.__file__)

vocab_fn="/home/jiangl1/11731_assign2/data_ted/vocab/en-beru.vocab"

if __name__=="__main__":
    vocab=utils.read_dict_from_pkl(vocab_fn)
    #print(vocab.src.id2word.values())
    print("done.")
