import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

TRAIN_DATA_PATH = f"data/sample_train.txt"
VALID_DATA_PATH = f"data/sample_val.txt"
TEST_DATA_PATH = f"data/sample_test.txt"
MODEL_LOG_DIR = f"model_weight"
TRAIN_BATCH_SIZE = 128
N_PROCS = 8
VALID_BATCH_SIZE = 64
lr = 0.005 #0.005 for DREAM-RNN and DREAM-CNN, 0.001 for DREAM-Attn
BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH))//TRAIN_BATCH_SIZE
BATCH_PER_VALIDATION = len(pd.read_csv(VALID_DATA_PATH))//TRAIN_BATCH_SIZE
SEQ_SIZE = 250
NUM_EPOCHS = 10 #80
CUDA_DEVICE_ID = 0

generator = torch.Generator()
generator.manual_seed(42)
#device = torch.device(f"cuda:{CUDA_DEVICE_ID}")


# Ensure CUDA is available before setting the device
device = torch.device(f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu")
#import different models, others can be found in the original repository uploaded by muntakim rafi. Here I use a CNN based architecture. In order to understand the syntax, please check the abstract classes defined in 
# in files inside the directory prixfixe

from prixfixe.autosome import (AutosomeCoreBlock,
                      AutosomeFinalLayersBlock)
from prixfixe.bhi import BHIFirstLayersBlock
from prixfixe.prixfixe import PrixFixeNet

first = BHIFirstLayersBlock(
            in_channels = 5,
            out_channels = 320,
            seqsize = 250,
            kernel_sizes = [9, 15],
            pool_size = 1,
            dropout = 0.2
        )

core = AutosomeCoreBlock(in_channels=first.out_channels,
                        out_channels =64,
                        seqsize=first.infer_outseqsize())

final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

model = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
)

from prixfixe.autosome import AutosomeDataProcessor

dataprocessor = AutosomeDataProcessor(
    path_to_training_data=TRAIN_DATA_PATH,
    path_to_validation_data=VALID_DATA_PATH,
    train_batch_size=TRAIN_BATCH_SIZE, 
    batch_per_epoch=BATCH_PER_EPOCH,
    train_workers=N_PROCS,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_train=True,
    shuffle_val=False,
    seqsize=SEQ_SIZE,
    generator=generator
)
next(dataprocessor.prepare_train_dataloader())

from prixfixe.autosome import AutosomeTrainer
trainer = AutosomeTrainer(
    model,
    device=torch.device(f"cuda:{CUDA_DEVICE_ID}"), 
    model_dir=MODEL_LOG_DIR,
    dataprocessor=dataprocessor,
    num_epochs=NUM_EPOCHS,
    lr = lr)

trainer.fit()



test_df = pd.read_csv(TEST_DATA_PATH, sep='\t')
test_df['rev'] = test_df['ID'].str.contains('\-_').astype(int)

model.load_state_dict(torch.load(f"{MODEL_LOG_DIR}/model_best_MSE.pth"))
model.eval()

def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'a': [1, 0, 0, 0],
            'g': [0, 1, 0, 0],
            'c': [0, 0, 1, 0],
            't': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}
    return [mapping[base] for base in seq]

# One-hot encode sequences and concatenate 'rev' column
encoded_seqs = []
Y_test_dev = []
Y_test_hk = []
# this might be potentially confusing-- i have retained the names of the columns from the original repository, and then changed names of my own data to match it. dev and hk just correspond to 2 columns with signal/ expression values

for i, row in tqdm(test_df.iterrows()):
    encoded_seq = one_hot_encode(row['Sequence'])
    rev_value = [row['rev']] * len(encoded_seq)
    encoded_seq_with_rev = [list(encoded_base) + [rev] for encoded_base, rev in zip(encoded_seq, rev_value)]
    encoded_seqs.append(encoded_seq_with_rev)
    Y_test_dev.append(row['Dev_log2_enrichment'])
    Y_test_hk.append(row['Hk_log2_enrichment'])

pred_expr_dev = []
pred_expr_hk = []

for seq in tqdm(encoded_seqs):
    pred = model(torch.tensor(np.array(seq).reshape(1,250,5).transpose(0,2,1), device = device, dtype = torch.float32)) # #can also predict on batches to speed up prediction
    pred_expr_dev.append(pred[0].detach().cpu().flatten().tolist())
    pred_expr_hk.append(pred[1].detach().cpu().flatten().tolist())

import numpy as np

# Convert to NumPy arrays and flatten
pred_expr_dev = np.array(pred_expr_dev).flatten()
pred_expr_hk = np.array(pred_expr_hk).flatten()
Y_test_dev = np.array(Y_test_dev).flatten()
Y_test_hk = np.array(Y_test_hk).flatten()

# Ensure they are the same length
min_len = min(len(pred_expr_dev), len(Y_test_dev))  # Find shortest array
pred_expr_dev = pred_expr_dev[:min_len]
Y_test_dev = Y_test_dev[:min_len]

min_len_hk = min(len(pred_expr_hk), len(Y_test_hk))
pred_expr_hk = pred_expr_hk[:min_len_hk]
Y_test_hk = Y_test_hk[:min_len_hk]

# Compute Pearson correlation
pearson_dev = np.corrcoef(pred_expr_dev, Y_test_dev)[0, 1]
pearson_hk = np.corrcoef(pred_expr_hk, Y_test_hk)[0, 1]

# Print results
print(f"Dev Pearson r: {pearson_dev:.4f}")
print(f"Hk Pearson r: {pearson_hk:.4f}")

np.savez("predictions.npz", pred_expr_dev=pred_expr_dev, pred_expr_hk=pred_expr_hk, 
         Y_test_dev=Y_test_dev, Y_test_hk=Y_test_hk)
