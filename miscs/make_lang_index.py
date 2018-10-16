import os
import sys
sys.path.append("../")
import utils
lang_list=['en','az','gl','be','tr','pt','ru']

lang_root_path="../lang"
if not os.path.exists(lang_root_path):
    os.makedirs(lang_root_path)

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
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        #   word freq less than something, just go away
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


'''
if __name__=="__main__":
    lang2id_dict={}
    id2lang_dict={}

    for id in range(0,len(lang_list)):
        lang=lang_list[id]
        lang2id_dict[lang]=id
        id2lang_dict[id]=lang

    lang2id_fn="../lang/lang2id.pkl"
    id2lang_fn="../lang/id2lang.pkl"

    utils.write_dict_to_pkl(lang2id_dict,lang2id_fn)
    utils.write_dict_to_pkl(id2lang_dict,id2lang_fn)
    print("done.")
'''


#   read object from dict
def read_dict_from_pkl(fn):
    import pickle

    with open(fn,"rb") as f:
        obj=pickle.load(f)
    return obj

#   vocab stats
if __name__=="__main__":
    for id in range(0, len(lang_list)):
        lang=lang_list[id]
        vocab_fn="/home/jiangl1/11731_assign2/data_ted/vocab/"+str(id)+".vocab"
        assert os.path.isfile(vocab_fn)
        vocab=read_dict_from_pkl(vocab_fn)
        print(lang,len(vocab.id2word))

    print("done.")