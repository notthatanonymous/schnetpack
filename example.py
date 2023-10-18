import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl

qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

qm9data = QM9(
    './qm9.db', 
    batch_size=100,
    num_train=1000,
    num_val=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
        trn.CastTo32()
    ],
    property_units={QM9.U0: 'eV'},
    num_workers=1,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=True, # set to false, when not using a GPU
    load_properties=[QM9.U0], #only load U0 property
)
qm9data.prepare_data()
qm9data.setup()

atomrefs = qm9data.train_dataset.atomrefs
print('U0 of hyrogen:', atomrefs[QM9.U0][1].item(), 'eV')
print('U0 of carbon:', atomrefs[QM9.U0][6].item(), 'eV')
print('U0 of oxygen:', atomrefs[QM9.U0][8].item(), 'eV')
