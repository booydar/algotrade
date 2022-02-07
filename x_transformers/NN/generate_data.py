from remote.variables import *
from remote.retrieval import *
from remote.listops import *
import numpy as np
import torch


def generate_data(generator, task_name, path='data', train_size=10_000, val_size=1_000, test_size=2_000, batch_size=32):
    Xs, ys = [], []
    total_size = train_size + test_size + val_size
    num_batches = total_size // batch_size * 3
    for _ in range(num_batches):
        X, y, _, _ = next(generator)
        Xs.append(X)
        ys.append(y)

    Xs = torch.vstack(Xs)
    ys = torch.vstack(ys)
    
    _, inds = np.unique(Xs, axis=0, return_index=True)
    inds = np.random.permutation(inds)
    
    Xs = Xs[inds][:total_size]
    ys = ys[inds][:total_size]
    
    np.save(f'{path}/{task_name}_train_X.npy', Xs[:train_size].cpu())
    np.save(f'{path}/{task_name}_train_y.npy', ys[:train_size].cpu())

    np.save(f'{path}/{task_name}_val_X.npy', Xs[train_size:train_size+val_size].cpu())
    np.save(f'{path}/{task_name}_val_y.npy', ys[train_size:train_size+val_size].cpu())

    np.save(f'{path}/{task_name}_test_X.npy', Xs[train_size+val_size:train_size+val_size+test_size].cpu())
    np.save(f'{path}/{task_name}_test_y.npy', ys[train_size+val_size:train_size+val_size+test_size].cpu())

    
class copy_generator:
    def __init__(self):
        self.src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        self.tgt_mask = torch.ones(BATCH_SIZE, DEC_SEQ_LEN+1).bool().cuda()
    
    def __next__(self):
        X = np.zeros([BATCH_SIZE, ENC_SEQ_LEN]).astype(int)
        y = np.zeros([BATCH_SIZE, DEC_SEQ_LEN+1]).astype(int)
        y[:, 0] = 1
        for i in range(BATCH_SIZE):
#             sequence_length = np.random.randint(1, ENC_SEQ_LEN)
            sequence_length = ENC_SEQ_LEN
            random_sequence = np.random.randint(2, NUM_TOKENS, sequence_length)
            
            X[i, :sequence_length] = random_sequence
            y[i, 1: 2 * sequence_length + 1] = np.concatenate([random_sequence] * 2)

        return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask        


class reverse_generator:
    def __init__(self):
        self.src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        self.tgt_mask = torch.ones(BATCH_SIZE, DEC_SEQ_LEN+1).bool().cuda()
    
    def __next__(self):
        X = np.zeros([BATCH_SIZE, ENC_SEQ_LEN]).astype(int)
        y = np.zeros([BATCH_SIZE, DEC_SEQ_LEN+1]).astype(int)
        y[:, 0] = 1
        for i in range(BATCH_SIZE):
            #             sequence_length = np.random.randint(1, ENC_SEQ_LEN)
            sequence_length = ENC_SEQ_LEN
            random_sequence = np.random.randint(2, NUM_TOKENS, sequence_length)
            
            X[i, :sequence_length] = random_sequence
            y[i, 1:sequence_length + 1] = random_sequence[::-1]


        return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask        


class retrieval_generator:
    def __init__(self):
        self.src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool()
        self.tgt_mask = torch.ones(BATCH_SIZE, DEC_SEQ_LEN+1).bool()
    
    def __next__(self):
        X = np.zeros([BATCH_SIZE, ENC_SEQ_LEN]).astype(int)
        y = np.zeros([BATCH_SIZE, DEC_SEQ_LEN+1]).astype(int)
        y[:, 0] = 10
        for i in range(BATCH_SIZE):
            X[i], y[i, 1:] = create_sequence(one_hot=False)


        return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask         


class listops_generator:
    def __init__(self, max_depth=2):
        self.src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        self.tgt_mask = torch.ones(BATCH_SIZE, DEC_SEQ_LEN+1).bool().cuda()
        self.max_depth = max_depth
    
    def __next__(self):
        X = np.zeros([BATCH_SIZE, ENC_SEQ_LEN]).astype(int)
        y = np.ones([BATCH_SIZE, 2]).astype(int) * 2
        for i in range(BATCH_SIZE):
            t = generate_tree(self.max_depth)
            tokens, value = to_tokens(t), to_value(t) 
            X[i, 0:len(tokens)], y[i, 1:] = tokens, value+2
            del t

        return torch.tensor(X), torch.tensor(y), self.src_mask, self.tgt_mask         


if __name__ == "__main__":
    task2gen = {'copy': copy_generator,
                'reverse': reverse_generator,
                'retrieval': retrieval_generator,
                'listops': listops_generator}

    generator = task2gen[TASK_NAME]()
    generate_data(generator, task_name=TASK_NAME, train_size=TRAIN_SIZE, test_size=TEST_SIZE, val_size=VAL_SIZE)  