import os
import sys
sys.path.append("../")
import utils

data_root_path="/home/jiangl1/11731_assign2/data_ted/"

type_list=["dev","train","test"]

#src_list=["en-aztr.aztr.txt"]
#dst_list=["all.en-aztr.aztr.txt"]
#src_or_tgt="src"

'''
src_list=["en-aztr.en.txt"]
dst_list=["all.en-aztr.en.txt"]
src_or_tgt="tgt"

src_list=["en-beru.beru.txt"]
dst_list=["all.en-beru.beru.txt"]
src_or_tgt="src"

src_list=["en-beru.en.txt"]
dst_list=["all.en-beru.en.txt"]
src_or_tgt="tgt"
'''

#src_list=["en-glpt.glpt.txt"]
#dst_list=["all.en-glpt.glpt.txt"]
#src_or_tgt="src"

src_list=["en-glpt.en.txt"]
dst_list=["all.en-glpt.en.txt"]
src_or_tgt="tgt"

def read_lines(fn):
    with open(fn,"r") as f:
        return f.readlines()

def write_lines(lines,fn):
    with open(fn,"w") as f:
        f.writelines(lines)


if __name__=="__main__":
    src_raw=src_list[0];dst_fn=os.path.join(data_root_path,dst_list[0])

    all_src=[]
    for t in type_list:
        cur_src_fn=os.path.join(data_root_path,t+"."+src_raw)
        cur_cor=read_lines(cur_src_fn)
        all_src+=cur_cor


    write_lines(all_src,dst_fn)

    print("done.")