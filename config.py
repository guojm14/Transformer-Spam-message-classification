import argparse

parser = argparse.ArgumentParser(description='message classification')

"----------------------------- General options -----------------------------"
parser.add_argument('--mod', default='train', type=str,
                    help='train or test')
parser.add_argument('--datadir', default='data', type=str,
                    help='Dataset')
parser.add_argument('--w2vmodel', default='sgns.baidubaike.bigram-char', type=str,
                    help='model of word2vec')
parser.add_argument('--snapshot', default=1, type=int,
help='How often to take a snapshot of the model (0 = never)')
