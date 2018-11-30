# -*- coding: utf-8 -*-
import gensim
import jieba
import numpy as np
from datetime import datetime 
a=datetime.now() 
b=datetime.now()
class sentence2vec(object):
    def __init__(self,w2vmodel):
        print 'loading word2vec model......'
        a=datetime.now() 
        self.model=gensim.models.KeyedVectors.load_word2vec_format(w2vmodel)
        b=datetime.now()
        print 'loading finished:'+str((b-a).microseconds)+'ms cost'
        self.vocab=self.model.vocab.keys()
        self.unknown=np.zeros(300)
    def w2v(self,word):
        #word = unicode(word,'utf-8')
        if word in self.vocab:
            return self.model[word]
        else:
            return self.unknown
 
    def __call__(self,sentence):
        outvec=np.array(map(self.w2v,jieba.lcut(sentence)))
        return outvec


if __name__ == '__main__':
    s2v=sentence2vec('sgns.weibo.bigram-char')
    st=open('data/train.txt').readlines()[0].split()[1]
    vec=s2v(st)
    print vec.shape
    vec1=s2v(st)
    print vec1.shape
