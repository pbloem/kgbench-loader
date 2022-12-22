import dgl
import torch.cuda
from dgl.contrib.data import load_data
from dgl.nn.pytorch import RelGraphConv

from torch import nn
import torch.nn.functional as F

import fire

import kgbench as kgb
from kgbench.util import d

"""
Use the DGL loader, and use DGL to run a (full batch) R-GCN.

Code adapted from the `entity.py` R-GCN example in DGL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

def go(name='aifb', final=False, lr=0.01, wd=5e-3, in_dim=256, h=16, num_bases=40, prune=False):

    # dataset = dgl.data.rdf.AIFBDataset()
    # g = dataset[0]
    # category = dataset.predict_category
    # num_classes = dataset.num_classes
    #
    # train_mask = g.nodes[category].data['train_mask']
    # test_mask = g.nodes[category].data['test_mask']
    # label = g.nodes[category].data['label']
    #
    # print(g.nodes)
    # print(type(train_mask), train_mask.size())

    dataset = kgb.load(name=name, torch=True, final=final, prune_dist=2 if prune else None).dgl(to32=True)

    g = dataset[0].int()

    if torch.cuda.is_available():
        g.to('cuda')

    num_rels = len(g.canonical_etypes)
    num_classes = dataset.num_classes

    print(f'Loaded {name}: {num_rels} relations, {num_classes} classes.')

    category = dataset.predict_category

    labels = g.nodes[category].data.pop('label') # note: not labels

    training_mask = g.nodes[category].data.pop('training_mask')
    withheld_mask = g.nodes[category].data.pop('withheld_mask')
    # -- note the use of training/withheld rather than train/test

    for cetype in g.canonical_etypes:
        g.edges[cetype].data['norm'] = dgl.norm_by_dst(g, cetype).unsqueeze(1)

    g = dgl.to_homogeneous(g, edata=['norm'])

    training_idx = training_mask.nonzero()
    withheld_idx = withheld_mask.nonzero()
    # -- nonzero() produces the indices of all the elements that are true

    training_labels = labels[training_idx].squeeze()
    withheld_labels = labels[withheld_idx].squeeze()

    # -- Note that this approach from the original script won't work
    #    >>> target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    #    We don't have a particular type for target nodes.

    model = RGCN(num_nodes=g.num_nodes(), in_dim=in_dim, h_dim=h, out_dim=dataset.num_classes, num_rels=num_rels, num_bases=num_bases)

    if torch.cuda.is_available():
        model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for epoch in range(50):
        logits = model(g)[training_idx].squeeze(1)
        loss = F.cross_entropy(logits, training_labels.to(torch.long))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == training_labels).sum() / training_labels.size(0)
        print(f'Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Accuracy {acc:.4f} ')

    acc = evaluate(g, withheld_idx, withheld_labels, model)
    print("Test accuracy {:.4f}".format(acc))

class RGCN(nn.Module):
    """
    NB: This is an _embedding_ RGCN. It's slightly different from the classic RGCN in rgcn.py.
    """

    def __init__(self, num_nodes, in_dim, h_dim, out_dim, num_rels, num_bases):

        super().__init__()

        self.emb = nn.Embedding(num_nodes, in_dim)
        # two-layer RGCN
        self.conv1 = RelGraphConv(in_dim, h_dim, num_rels, regularizer='basis', num_bases=num_bases, self_loop=True)
        self.conv2 = RelGraphConv(h_dim, out_dim, num_rels, regularizer='basis', num_bases=num_bases, self_loop=True)

    def forward(self, g):

        x = self.emb.weight
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata['norm']))

        h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata['norm'])
        return h

def evaluate(g, withheld_idx, labels, model):

    model.eval()
    with torch.no_grad():
        logits = model(g)[withheld_idx].squeeze(1)

    acc = (logits.argmax(dim=1) == labels).sum() / labels.size(0)

    return acc.item()

if __name__ == '__main__':
    fire.Fire(go)