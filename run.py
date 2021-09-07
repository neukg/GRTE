import argparse
from main import *

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--cuda_id', default="0", type=str)
parser.add_argument('--dataset', default='WebNLG', type=str)
parser.add_argument('--rounds', default=4, type=int)
parser.add_argument('--train', default="train", type=str)

parser.add_argument('--batch_size', default=6, type=int)
parser.add_argument('--test_batch_size', default=6, type=int)
parser.add_argument('--file_id', default="99", type=str)
parser.add_argument('--learning_rate', default=3e-5, type=float)
parser.add_argument('--num_train_epochs', default=50, type=int)
parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
parser.add_argument('--bert_vocab_path', default="./pretrained/bert-base-cased/vocab.txt", type=str)
parser.add_argument('--bert_config_path', default="./pretrained/bert-base-cased/config.json", type=str)
parser.add_argument('--bert_model_path', default="./pretrained/bert-base-cased/pytorch_model.bin", type=str)
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--min_num', default=1e-7, type=float)
parser.add_argument('--base_path', default="./dataset", type=str)

args = parser.parse_args()

if args.train=="train":
    train(args)
else:
    test(args)