from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataloader import *
import tensorflow as tf
import attention_layer
import ffn_layer
import model_utils
from tqdm import tqdm
hiddensize=32
dropout=0.3
hiddenlayers=3
numhead=4
attentiondropout=0.1
filtersize=256
reludropout=0.1
class model(object):
  def __init__(self,train,batch_size):
    self.train=train
    self.batch_size=batch_size
    self.encoder_stack=EncoderStack(train)
    self.embedding_layer=tf.layers.Dense(hiddensize, name="embedding")
    self.outfc=tf.layers.Dense(2, name="embedding")
  def __call__(self,inputs,padnum,pos):
    initializer = tf.variance_scaling_initializer(
        1, mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
      attention_bias = model_utils.get_padding_bias(padnum)
      encoderout=self.encode(inputs, attention_bias,padnum,pos)
    return encoderout
  def encode(self, inputs, attention_bias,pad_num,pos):
    """Generate continuous representation for inputs.
    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      #length=inputs.shape[1]
      embedded_inputs = tf.reshape(self.embedding_layer(tf.reshape(inputs,[-1,300])),[self.batch_size,-1,hiddensize])
      inputs_padding = model_utils.get_padding(pad_num)

      with tf.name_scope("add_pos_encoding"):
        length = embedded_inputs.shape[1]
        #pos_encoding = model_utils.get_position_encoding(length, hiddensize)
        #print(embedded_inputs)
        #print(pos_encoding)
        #encoder_inputs=embedded_inputs
        encoder_inputs = embedded_inputs + pos

     # if self.train:
     #   encoder_inputs = tf.nn.dropout(
     #       encoder_inputs, 1 - dropout)

      encoderout= self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)
      out=self.outfc(tf.reshape(encoderout[:,0,:],[-1,hiddensize]))
      return out
class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, train):
    self.layer = layer
    self.postprocess_dropout = dropout
    self.train = train

    # Create normalization layer
    self.layer_norm = LayerNormalization(hiddensize)

  def __call__(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if self.train:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
    return x + y


class EncoderStack(tf.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, train):
    super(EncoderStack, self).__init__()
    self.layers = []
    for _ in range(hiddenlayers):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          hiddensize, numhead,
          attentiondropout, train)
      feed_forward_network = ffn_layer.FeedFowardNetwork(
          hiddensize, filtersize,
          reludropout, train, True)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer,train),
          PrePostProcessingWrapper(feed_forward_network,train)])

    # Create final layer normalization layer.
    self.output_normalization = LayerNormalization(hiddensize)

  def call(self, encoder_inputs, attention_bias, inputs_padding):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer.
        [batch_size, 1, 1, input_length]
      inputs_padding: P

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)
    
    #return encoder_inputs
    return self.output_normalization(encoder_inputs)
def test():
    model_dir='model2/'
    logfile='test2.log'
    fop=open(logfile,'w')
    #prepare data
    
    dataline=open('data/train.txt').readlines()
    datalength=len(dataline)
    testdata=dataline[:int(datalength/5)]
    vecmodel=word2vec.sentence2vec('sgns.weibo.bigram-char')

    inputdata=tf.placeholder(tf.float32,[1,None,300])
    inputpadding=tf.placeholder(tf.float32,[1,None])
    pos=tf.placeholder(tf.float32,[None,32])
    inputlabel=tf.placeholder(tf.int32,[1])
    classifier=model(True,1)
    outlabel=tf.argmax(classifier(inputdata,inputpadding,pos),1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=4)
    model_file=tf.train.latest_checkpoint(model_dir)
    saver.restore(sess,model_file)
    sum=0
    accu=0
    for i in range(len(testdata)):
         vec=np.array([vecmodel(''.join(testdata[i].split()[1:]))])
         padnum=np.ones((1,vec.shape[0]))
         logit =sess.run(outlabel,feed_dict={inputdata:vec,inputpadding:padnum,pos:model_utils.get_position_encoding(len(vec),32)})[0]
         label =testdata[i].split()[0]
         print(logit,label,int(logit)==int(label))
         fop.write(str(logit)+' '+str(label)+'\n')
         sum+=1
         if logit==label:
             accu+=1
    print(accu/sum)
   
def train():
    #config
    batch_size=4
    lr=0.0005
    model_dir='model2/'
    logfile='second.log'
    fop=open(logfile,'w')
    #prepare data
   
    dataline=open('data/train.txt').readlines()
    datalength=len(dataline)
    traindata=dataline[int(datalength/5):]
    print(len(traindata))
    vecmodel=word2vec.sentence2vec('sgns.weibo.bigram-char')
    
    a=dataloader(traindata,vecmodel,batch_size)
    a.start()
    
    #build model
    inputdata=tf.placeholder(tf.float32,[batch_size,None,300])
    inputpadding=tf.placeholder(tf.float32,[batch_size,None])
    pos=tf.placeholder(tf.float32,[None,32])
    inputlabel=tf.placeholder(tf.int32,[batch_size])
    classifier=model(True,batch_size)
    outlogit=classifier(inputdata,inputpadding,pos)
    loss=tf.losses.softmax_cross_entropy(tf.one_hot(inputlabel,2),outlogit)
    print(1)
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(max_to_keep=0)

    print('build finished')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model_file=tf.train.latest_checkpoint(model_dir)
    saver.restore(sess,model_file)

    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    
    #train step
    for step in tqdm(range(26000,400000)):
        data,label,padding= a.getdata()
        #data=np.zeros((3,20,300))
        #padding=np.ones((3,20))
        length=data.shape[1]
        trainloss,_=sess.run([loss,train_op],feed_dict={inputdata:data,inputpadding:padding,pos:model_utils.get_position_encoding(length,32),inputlabel:label})
        if step%100==0:
            print('loss:'+str(trainloss))
            fop.write('loss:'+str(trainloss)+'\n')
        if step%1000==0:
            saver.save(sess,model_dir+'/transform.ckpt',global_step=step)

test()
