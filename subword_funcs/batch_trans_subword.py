'''
    10/22/2018: Batch transform the original paragraph into subword representations.
'''

import os
import sys
sys.path.append("../")
import utils


file_list=["en-aztr.aztr.txt","en-aztr.en.txt",
           "en-beru.beru.txt","en-beru.en.txt",
           "en-glpt.glpt.txt","en-glpt.en.txt"]


#file_list=["en-az.az.txt","en-be.be.txt","en-gl.gl.txt","en-az.en.txt","en-be.en.txt","en-gl.en.txt"]

#type_list=["dev","train","test","all"]
type_list=["all"]

org_root_path="/home/jiangl1/11731_assign2/data_ted/"
dst_root_path="/home/jiangl1/11731_assign2/data_ted.subword/"

def get_lang_from_raw_file(raw_file):
    tmp_str=raw_file.split(".")
    if tmp_str[-2]=="az":
        return "aztr"
    if tmp_str[-2]=="be":
        return "beru"
    if tmp_str[-2]=="gl":
        return "glpt"

    return tmp_str[-2]

def handle_a_file(raw_file):
    lang = get_lang_from_raw_file(raw_file)
    vocab_full_fn=os.path.join("/home/jiangl1/11731_assign2/data_ted/vocab/",lang+".subword.vocab")
    assert os.path.isfile(vocab_full_fn)

    for t in type_list:
        raw_t_file=t+"."+raw_file
        print(raw_t_file)
        org_full_fn=os.path.join(org_root_path,raw_t_file)
        dst_full_fn=os.path.join(dst_root_path,raw_t_file)
        assert os.path.isfile(org_full_fn)

        cmd="python segment_chars.py --input %s --shortlist 1500 -n 4 --output %s --vocab %s"%(org_full_fn,dst_full_fn,vocab_full_fn)
        print(cmd)
        os.system(cmd)

    return

if __name__=="__main__":
    for raw_file in file_list:
        handle_a_file(raw_file)
    print("done.")
