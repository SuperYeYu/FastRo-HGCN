from __future__ import division
from __future__ import print_function

import random
import argparse
import tools
import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorchtools import EarlyStopping
from utils import accuracy, load_ACM_data
from models import FastRo_HGCN
from sklearn.metrics import f1_score
import time
import psutil
import os

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0000)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--L_hidden_dim', type=int, default=64)
parser.add_argument('--num-layers', type=int, default=1)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--model', type=str, default="FastRo-HGCN")
parser.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is ACM.')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
features_list, adjM, type_mask, labels, train_val_test_idx,e = load_ACM_data()
labels = torch.LongTensor(labels)

features_list = [torch.FloatTensor(features) for features in features_list]

e_weight=e
e_weight=torch.FloatTensor(e_weight)
g = dgl.DGLGraph(adjM)

train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']

in_dims = [features.shape[1] for features in features_list]

print("data load finish")

def train(model, optimizer):
    t1=time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features_list, g, type_mask,e_weight)
    logp = F.log_softmax(output, 1)
    loss_train = F.nll_loss(logp[train_idx], labels[train_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    t2 = time.time()
    model.eval()
    with torch.no_grad():
        output = model(features_list, g, type_mask,e_weight)

        logp = F.log_softmax(output, 1)
        loss_val = F.nll_loss(logp[val_idx], labels[val_idx])
        loss_test = F.nll_loss(logp[test_idx], labels[test_idx])
    return loss_val,loss_test, model,t2-t1


def compute_test(model):
    model.eval()
    test_logits = []
    with torch.no_grad():
        output = model(features_list, g, type_mask,e_weight)
        test_logits = output[test_idx]
        embeddings = test_logits.detach().cpu().numpy()
        acc,test_micro_f1, test_macro_f1 = score(test_logits, labels[test_idx])
        print('micro=',test_micro_f1*100,' macro=',test_macro_f1*100)
        print(test_micro_f1*100)
        svm_macro, svm_micro, nmi, ari = tools.evaluate_results_nc(embeddings, labels[test_idx].cpu().numpy(), int(labels.max()) + 1)
    return svm_macro, svm_micro, nmi, ari

svm_macro_avg = np.zeros((7, ), dtype=np.float)
svm_micro_avg = np.zeros((7, ), dtype=np.float)
nmi_avg = 0
ari_avg = 0
for cur_repeat in range(args.repeat):
    if args.model == "FastRo-HGCN":
        model = FastRo_HGCN(input_dim=in_dims,l_dim=args.L_hidden_dim, hidden_dim=args.hidden,
                    layers=args.num_layers, num_classes=int(labels.max()) + 1, dropout=args.dropout)
        print('# model parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format(args.save_postfix))
    a=0
    b=0
    for epochs in range(args.epochs):
        b=b+1
        start=time.time()
        val_loss,test_loss, net,a2 = train(model, optimizer)
        end=time.time()
        print(u'Memory:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        a=a+a2
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break


    print('\ntesting...and average time=',a/b)
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args.save_postfix)))
    svm_macro, svm_micro, nmi, ari = compute_test(net)
    svm_macro_avg = svm_macro_avg + svm_macro
    svm_micro_avg = svm_micro_avg + svm_micro
    nmi_avg += nmi
    ari_avg += ari

svm_macro_avg = svm_macro_avg / args.repeat
svm_micro_avg = svm_micro_avg / args.repeat
nmi_avg /= args.repeat
ari_avg /= args.repeat
print('---\nThe average of {} results:'.format(args.repeat))
print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
print('NMI: {:.6f}'.format(nmi_avg))
print('ARI: {:.6f}'.format(ari_avg))
print('all finished')

with open('log-statistics.txt', 'a+') as f:
    f.writelines('\n' + 'Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]) + '\n' +
                 'Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]) + '\n')

