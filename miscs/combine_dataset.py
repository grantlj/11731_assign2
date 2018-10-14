'''
    10/13/2018: Combine the parallel language corpus dataset from multiple languages for training.
'''

import os
import sys
import random
import pickle
sys.path.append("../")

data_root_path="../data_ted/"
assert os.path.exists(data_root_path)

src_list=['gl','pt']
tgt='en'

new_src_name="".join(src_list)
split_list=['dev','test','train']

def read_lines(fn):
    with open(fn,"r") as f:
        all_lines=f.readlines()
    all_lines=[x.replace("\n","") for x in all_lines]
    all_lines=[x.replace("\r","") for x in all_lines]

    return all_lines

def read_pairs(src,tgt,split):
    src_raw_fn="%s.%s-%s.%s.txt"%(split,tgt,src,src)
    full_src_raw_fn=os.path.join(data_root_path,src_raw_fn)
    assert os.path.isfile(full_src_raw_fn)

    tgt_raw_fn = "%s.%s-%s.%s.txt" % (split, tgt, src, tgt)
    full_tgt_raw_fn = os.path.join(data_root_path, tgt_raw_fn)
    assert os.path.isfile(full_tgt_raw_fn)

    src_lines=read_lines(full_src_raw_fn)
    tgt_lines=read_lines(full_tgt_raw_fn)

    ret=[]

    for src_sent,tgt_sent in zip(src_lines,tgt_lines):
        ret.append([(src,src_sent),(tgt,tgt_sent)])

    return ret

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

def write_lines(all_lines,fn):
    with open(fn,"w") as f:
        f.writelines(all_lines)
    return

def write_pairs_to_new_file(all_pairs):
    dst_src_raw_fn = "%s.%s-%s.%s.txt" % (split, tgt, new_src_name, new_src_name)
    dst_src_full_fn=os.path.join(data_root_path,dst_src_raw_fn)
    dst_tgt_raw_fn="%s.%s-%s.%s.txt" % (split, tgt, new_src_name, tgt)
    dst_tgt_full_fn=os.path.join(data_root_path,dst_tgt_raw_fn)

    dst_type_fn="%s.%s-%s.type.pkl" % (split, tgt, new_src_name)
    dst_type_full_fn=os.path.join(data_root_path,dst_type_fn)

    pair_index=[]
    src_lines=[]
    tgt_lines=[]
    for pair_meta in all_pairs:
        src_meta=pair_meta[0];tgt_meta=pair_meta[1]
        src_type=src_meta[0];src_text=src_meta[1]
        tgt_type=tgt_meta[0];tgt_text=tgt_meta[1]

        pair_index.append({'src':src_type,'tgt':tgt_type})
        if not src_text[-1]=="\n":
            src_text+="\n"
        if not tgt_text[-1]=="\n":
            tgt_text+="\n"

        src_lines.append(src_text)
        tgt_lines.append(tgt_text)

    write_lines(src_lines,dst_src_full_fn)
    write_lines(tgt_lines,dst_tgt_full_fn)

    write_dict_to_pkl(pair_index,dst_type_full_fn)

    return


def handle_a_split(split):
    all_pairs=[]
    for src in src_list:
        print("Handling split: ",split," soure language: ",src,"...")
        cur_pairs=read_pairs(src,tgt,split)
        all_pairs+=cur_pairs

    random.shuffle(all_pairs)
    write_pairs_to_new_file(all_pairs)

    return

if __name__=="__main__":
    for split in split_list:
        handle_a_split(split)

    print("done.")