from  scipy import io
import os
import threading
import Queue
import numpy as np
from PIL import Image
import random
import word2vec
def getpadding(length,bs):
    padding=np.zeros((bs,max(length)))
    for i in range(bs):
        for j in range(max(length)):
            if j < length[i]:
                padding[i][j]=1
    return padding

def padding(data,bs,length):
    temp=[]
    for i in range(bs):
        temp.append(np.append(data[i],np.zeros((max(length)-length[i],300)),axis=0))
    return np.array(temp)
class dataloader(threading.Thread):
    def __init__(self,dataline,model,batchsize=32,t_name='heihei'):
        threading.Thread.__init__(self, name=t_name)  
        self.dataline=dataline
        self.model=model
        self.dataqueue=Queue.Queue(maxsize=10)
        self.bs=batchsize
        self.on=True
        self.index=0
        self.datalength=len(self.dataline)
        self.epoch=0
        self.which=range(self.datalength)
        random.shuffle(self.which)
        print 'inited'
    def run(self):
        while(self.on):            
            data=[]
            label=[]
            length=[]
            for i in xrange(self.bs):
                item=self.dataline[self.which[self.index]]
                stlabel=item.split()[0]
                vec=self.model(''.join(item.split()[1:]))
                lengthvec=vec.shape[0]
                length.append(lengthvec)
                data.append(vec)
                label.append(stlabel)
                self.index+=1
                if self.index==self.datalength:
                    random.shuffle(self.which)
                    self.index=0
                    self.epoch+=1
            item=(padding(data,self.bs,length),np.array(map(int,label)),getpadding(length,self.bs))
            self.dataqueue.put(item)
    def getdata(self):
        return self.dataqueue.get()
    def close(self):
        self.on=False
def testcode():
    dataline=open('data/train.txt').readlines()
    model=word2vec.sentence2vec('sgns.weibo.bigram-char')
    a=dataloader(dataline,model,batchsize=3)
    a.start()
    data,label,padding= a.getdata()
    print data.shape
    print label
    print padding
    a.close()
