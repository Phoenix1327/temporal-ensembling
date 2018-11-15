import torch
import torch.nn as nn
import config
from temporal_ensembling import train
from models import BaseCNN
from utils import savetime, save_exp
import pdb

# metrics
accs         = []
accs_best    = []
losses       = []
sup_losses   = []
unsup_losses = []
idxs         = []


ts = savetime()
cfg = vars(config)

#pdb.set_trace()
for i in xrange(cfg['n_exp']):
    model = BaseCNN(cfg['batch_size'], cfg['std'])
    seed = cfg['seeds'][i]
    acc, acc_best, l, sl, usl, indices = train(model, seed, **cfg)
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(indices)

print 'saving experiment'    

#save_exp(ts, losses, sup_losses, unsup_losses,
#         accs, accs_best, idxs, **cfg)

