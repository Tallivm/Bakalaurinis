#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import manual_seed, autograd, cuda, device, LongTensor, FloatTensor, from_numpy, tensor, float32, int32, \
    long, zeros
from torch.nn import Module, Linear, ReLU, Dropout, Embedding, NLLLoss
from torch.nn.functional import log_softmax
from torch.optim import Adam
from torch import max as tmax
from torch import exp as texp
from torch import where as twhere
from torch import cat
from torch.utils.data import Dataset, DataLoader
manual_seed(420)

device = device('cuda' if cuda.is_available() else 'cpu')
import numpy as np
    
import random
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


embedding_size = 200
cf = 400
h  = 50
kc = 6
kp = 2
lr = 0.0001
num_epochs = 1
minibatchSize = 512
l2_reg_rate = 0.00001

rng = random.Random(33)


# In[ ]:


class DeepVS(Module):
    def __init__(self, vocab_size, embedding_dim, cf, h, kc, kp):
        super().__init__() 

        self.embeddings_context = Embedding(vocab_size, embedding_dim) 
        self.linear1 = Linear(( (kc+kp)*3 + kp ) * embedding_dim, cf)
        self.linear2 = Linear(cf, h)
        self.linear3 = Linear(h, 2)
        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)
        self.dropout = Dropout(.5, inplace=False)
        
    def forward(self, inputs, mask):
        mbSize = inputs.size()[0]
        
        #print('to embeddings:', inputs.view(-1, inputs.size()[2]).shape)
        embeddings = self.embeddings_context(inputs.view(-1, inputs.size()[2]))
        #embeddings = self.embeddings_context(inputs)
        #print('embeddings:', embeddings.shape)
        # first layer
        embeddings.requires_grad_(True)
        #print('to 1 layer', embeddings.view(embeddings.size()[0], -1).shape)
        out = self.linear1(embeddings.view(embeddings.size()[0], -1))
        #print('from 1 layer:', out.shape)
        out = self.relu1(out)
#         print('after relu:', out.shape)
        out = out.view(mbSize, int(out.size()[0]/mbSize), out.size()[1]) # puts minibatch dimension back
        #print('after view:', out.shape)
        #print('added mask shape:', mask.view(mask.size()[0], mask.size()[1], 1).shape)
        out = out + mask.view(mask.size()[0], mask.size()[1], 1)
        #print('suming with mask:', out.shape)
        out = tmax(out, dim=1)[0]
        #print('after max:', out.shape)
        out = self.dropout(out)
#         print('after dropout:', out.shape)
        
        # second layer
        out = self.linear2(out)
#         print('from 2 layer:', out.shape)
        out = self.relu2(out)
#         print('after relu:', out.shape)
        out = self.dropout(out)
#         print('after dropout:', out.shape)
        
        # third layer
        out = self.linear3(out) 
#         print('from 3 layer:', out.shape)
        log_probs = log_softmax(out, dim=1)
#         print('log_probs:', out.shape)
        return log_probs

class MolDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data[0]
        self.data.index = pd.CategoricalIndex(self.data[0].astype('category')).astype(int)
        self.data = self.data.drop(columns=[0])
        self.mask = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.data.index.unique())
    
    def __getitem__(self, mol_index):
        mol = tensor(self.data.loc[mol_index, 2:].values, dtype=long).to(device)  # 2D array
        msk = tensor(self.mask.loc[mol_index].values, dtype=long).to(device) # 2D array
        label = tensor(self.data.loc[mol_index, 1].values[0], dtype=long).to(device)
        return mol, label, msk


# In[ ]:


def load_restrictions(protein_name):
    with open('../data/protein.groups', 'r') as f:
        groups = f.readlines()
    groups = [x.strip('\n').split(',') for x in groups]
    groups = [x for x in groups if protein_name in x][0]
    
    with open('../data/protein.cross_enrichment', 'r') as f:
        cross_enrichment = f.readlines()
    cross_enrichment = [x.strip('\n').split(',') for x in cross_enrichment]
    cross_enrichment = [x for x in cross_enrichment if x[0] == protein_name]
    if cross_enrichment:
        groups.extend(cross_enrichment[0])
    return set(groups)

def import_data_to_df(proteins):
    df = pd.DataFrame()
    for pname in proteins:
        print(f'Importing {pname}...')
        file_name = f'../data/dud_vinaout_deepvs/{pname}.deepvs'
        mol_names = []
        atom_vectors = []
        with open(file_name, 'r') as f:
            for line in f:
                if line.startswith('@'):
                    name, cls = line.strip('\n').split(',')
                    if name not in mol_names:
                        mol_names.append(name)
                        while True:
                            line = f.readline()
                            line = line.strip('\n')
                            if line:
                                atom_vectors.append([name]+[int(cls)]+line.split(','))
                            else: break   # empty line shows the end of molecule
        df = df.append(atom_vectors)
    l = kc+kp
    df = df.reset_index(drop=True)

    df.iloc[:,0] = df.iloc[:,0].astype('category').cat.codes
    df.loc[:,2:l] = df.loc[:,2:l] + '_atm'
    df.loc[:,l+2:l*2+1] = df.loc[:,l+2:l*2+1] + '_dis'
    df.loc[:,l*2+2:l*3+1] = df.loc[:,l*2+2:l*3+1] + '_chr'
    df.loc[:,l*3+2:] = df.loc[:,l*3+2:] + '_amino'
    return df    

def create_context_dictionary(df, df_test):
    context_dict = {'MSK':0}
    train_uniques = set([x for l in [df[col].unique().tolist() for col in df if col not in [0,1]] for x in l])
    test_uniques = set([x for l in [df_test[col].unique().tolist() for col in df_test if col not in [0,1]] for x in l])
    uniques = train_uniques|test_uniques
    for i, word in enumerate(uniques, start=1):
        context_dict[word] = i
    return context_dict

def translate_to_ix(df, context_dict):
    for col in df:
        if col not in [0,1]:
            df[col] = df[col].map(context_dict).fillna(0)
    return df

def count_enrichment(scores, cls, perc):
    n = round(len(scores)*perc)
    _, indices = scores.sort(descending=True)
    cls_n = cls[indices][:n]
    enrichment = sum(cls_n).float() / len(cls_n)
    enrichment_factor = enrichment * len(cls) / sum(cls).float()
    return enrichment_factor


# In[ ]:


proteins = ['ace', 'ache', 'ada', 'alr2', 'ampc', 'ar', 'cdk2','comt', 'cox1', 
            'cox2', 'dhfr', 'egfr', 'er_agonist', 'er_antagonist', 'fgfr1', 'fxa', 
            'gart', 'gpb', 'gr', 'hivpr', 'hivrt', 'hmga', 'hsp90', 'inha', 'mr', 
            'na', 'p38', 'parp', 'pde5', 'pdgfrb', 'pnp', 'ppar', 'pr', 'rxr', 'sahh', 
            'src', 'thrombin', 'tk', 'trypsin', 'vegfr2']


# In[ ]:


test_protein = 'tk'
restricted_proteins = load_restrictions(test_protein)
train_proteins = [p for p in proteins if p not in restricted_proteins]
#train_proteins = ['ada', 'ace']
#train_proteins = train_proteins[:10]

# load
print('Loading train data...')
df = import_data_to_df(train_proteins)
print('Loading test data...')
df_test = import_data_to_df([test_protein])

# translate
print('Translating...')
context_dict = create_context_dictionary(df, df_test)
vocab_size = len(context_dict)+1
df = translate_to_ix(df, context_dict)
df_test = translate_to_ix(df_test, context_dict)

# mask
print('Masking...')
v_len = (kc+kp)*3 + kp
max_mol = max(df.groupby(0).count().max().max(), df_test.groupby(0).count().max().max())
c = pd.Series(df.groupby(0).count().iloc[:, 0], name='mol_sizes')
c_test = pd.Series(df_test.groupby(0).count().iloc[:, 0], name='mol_sizes')
df_padding = [[[i]+[0]*(v_len+1)]*(max_mol-x) for i, x in enumerate(c)]
df = df.append([x for l in df_padding for x in l], ignore_index=True)
df_test_padding = [[[i]+[0]*(v_len+1)]*(max_mol-x) for i, x in enumerate(c_test)]
df_test = df_test.append([x for l in df_test_padding for x in l], ignore_index=True)
df_mask = pd.DataFrame(np.zeros([len(df), 1]))
df_mask.iloc[len(df) - len([x for l in df_padding for x in l]):] = -999
df_test_mask = pd.DataFrame(np.zeros([len(df_test), 1]))
df_test_mask.iloc[len(df_test) - len([x for l in df_test_padding for x in l]):] = -999

del df_padding, df_test_padding

# to pytorch Datasets
print('Converting to PyTorch datasets...')
train_loader = DataLoader(MolDataset([df, df_mask]), minibatchSize, shuffle=True)
test_loader = DataLoader(MolDataset([df_test, df_test_mask]), minibatchSize, shuffle=False)

# weights for loss function
class1 = round(sum(df[1]) / len(df[1]), 3)
class0 = 1-class1
loss_weights = tensor((class1, class0)).to(device)

del df, df_test, df_mask, df_test_mask
# In[ ]:


print('Initializing model...')
model = DeepVS(vocab_size, embedding_size, cf, h, kc, kp)
model.to(device)
loss_function = NLLLoss()
# loss_function = NLLLoss(weight=loss_weights)
optimizer = Adam(model.parameters(), lr, weight_decay=l2_reg_rate)


# In[ ]:


print('Training...')
for epoch in range(1, num_epochs+1):
    #total_loss = 0
    TP, TN, FP, FN = zeros(1).to(device), zeros(1).to(device), zeros(1).to(device), zeros(1).to(device)
    all_scores, all_cls = None, None
    model.train()
    #print('Loading data from dataloader...')
    for cmplx, cls, msk in train_loader:
        print(cmplx.shape, cls.shape, msk.shape)
        # important for optimizer to reset
        #print('Loaded I guess')
        model.zero_grad()
        #print('Made zero_grad')
        # Run the forward pass
        log_probs = model.forward(cmplx, msk)
        #print('Got log_probs')
        # Compute loss and update model 
        loss = loss_function(log_probs, cls)
        loss.backward()
        optimizer.step()
        #total_loss += loss.data.item()    
        
        _, preds = tmax(texp(log_probs), 1)
        confusion_vector = preds + cls*2
        TP += sum(twhere(confusion_vector==3, tensor(1).to(device), tensor(0).to(device)))
        TN += sum(twhere(confusion_vector==0, tensor(1).to(device), tensor(0).to(device)))
        FP += sum(twhere(confusion_vector==1, tensor(1).to(device), tensor(0).to(device)))
        FN += sum(twhere(confusion_vector==2, tensor(1).to(device), tensor(0).to(device)))
        
        scores = texp(log_probs)[:, -1]
        if all_scores is None:
            all_scores = scores
            all_cls = cls
        else:
            all_scores = cat([all_scores, scores])
            all_cls = cat([all_cls, cls])
    enrichment_2 = count_enrichment(all_scores, all_cls, 0.02)
    enrichment_20 = count_enrichment(all_scores, all_cls, 0.2)
    print("-"*30)  
    print(f"epoch = {epoch}") 
    print(f'Train confusions: TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    if sum(TP, FP) != 0: print(f'Precision: {TP/(TP+FP)}')
    else: print('! Zero precision because no positives !') 
    print(f'EF2%: {enrichment_2}; EF20%: {enrichment_20}')
    # sets model to eval (needed to use dropout in eval mode)
    model.eval()
    # Test model after each epoch
    #total_loss_test = 0
    TP, TN, FP, FN = zeros(1).to(device), zeros(1).to(device), zeros(1).to(device), zeros(1).to(device)
    all_scores, all_cls = None, None
    
    for cmplx_test, cls_test, msk_test in test_loader:
        # Run the forward pass
        outputs = model.forward(cmplx_test, msk_test)
        loss_test = loss_function(outputs, cls_test)
        #total_loss_test += loss_test.data.item()
        
        _, preds = tmax(texp(outputs), 1)
        confusion_vector = preds + cls_test*2
        TP += sum(twhere(confusion_vector==3, tensor(1).to(device), tensor(0).to(device)))
        TN += sum(twhere(confusion_vector==0, tensor(1).to(device), tensor(0).to(device)))
        FP += sum(twhere(confusion_vector==1, tensor(1).to(device), tensor(0).to(device)))
        FN += sum(twhere(confusion_vector==2, tensor(1).to(device), tensor(0).to(device)))
        
        scores = texp(outputs)[:,-1]
        if all_scores is None:
            all_scores = scores
            all_cls = cls
        else:
            all_scores = cat([all_scores, scores])
            all_cls = cat([all_cls, cls])
    enrichment_2 = count_enrichment(all_scores, all_cls, 0.02)
    enrichment_20 = count_enrichment(all_scores, all_cls, 0.2)        
    #print(f"mean training loss = {total_loss/minibatchSize}; mean test loss = {total_loss_test/minibatchSize}")
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    if TP+FP != 0: print(f'Precision: {TP/(TP+FP)}')
    else: print('! Zero precision because no positives !')    
    print(f'EF2%: {enrichment_2}; EF20%: {enrichment_20}')


# ### Old codes

# In[ ]:


# def import_data(train_proteins):
#     MolName, MolClass, MolData = [], [], [] 
#     for pname in train_proteins:
#         print(f'Importing {pname}...')
#         file_name = f'../data/dud_vinaout_deepvs/{pname}.deepvs'
        
#         with open(file_name, 'r') as f:
#             data = f.readlines()
#         data = [line.strip('\n') for line in data]
#         mol_vectors = []
#         for line in data:
#             if line.startswith('@'):
#                 if mol_vectors: MolData.append(mol_vectors)
#                 mol_vectors = []
#                 name, cls = line.split(',')
#                 MolName.append(name)
#                 MolClass.append(cls)
#             elif line:
#                 line = line.split(',') 
#                 l = kc+kp
#                 atoms =     [f'{x}_atm' for x in line[:l]]
#                 distances = [f'{x}_dis' for x in line[l:l*2]]
#                 charges =   [f'{x}_chr' for x in line[l*2:l*3]]
#                 amino =     [f'{x}_amn' for x in line[l*3:]]
#                 line = atoms + distances + charges + amino
#                 mol_vectors.append(line)
#         MolData.append(mol_vectors) 
#     combined = list(zip(MolData, MolName, MolClass))
#     rng.shuffle(combined)
#     MolData, MolName, MolClass = zip(*combined)
#     del combined
#     return MolName, MolClass, MolData

# def create_context_dictionary(MolData):
#     context_to_ix = {'UNK':0}
#     ix = 1
#     for ligand in MolData:
#         for context in ligand:
#             for position in context:
#                 if position not in context_to_ix:
#                     context_to_ix[position] = ix
#                     ix += 1    
#     return context_to_ix

# def translate_to_ix(context_to_ix, MolData):
#     unk_idx = context_to_ix['UNK']
#     temp_MolData = []
#     for ligand in MolData:
#         temp_ligand = []
#         temp_MolData.append(temp_ligand)
#         for context in ligand:
#             temp_context = []
#             temp_ligand.append(temp_context)
#             for position in context:
#                 temp_context.append(context_to_ix.get(position, unk_idx)) 
                
#     return temp_MolData

# def prepare_minibatches(temp_MolData, MolClass, minibatchSize):
#     new_molData   = []
#     current_Data  = []
#     current_Class = []

#     for m, c in zip(temp_MolData, MolClass):
#         current_Data.append(m)
#         current_Class.append(c)
#         if len(current_Data) == minibatchSize:
#             new_molData.append([current_Data, current_Class])
#             current_Data = []
#             current_Class = []

#     if len(current_Data) > 0:
#         new_molData.append([current_Data, current_Class])

#     # creates the fake molecule contexts:  
#     fake = [0]  * len(new_molData[0][0][0][0]) # indexes meaning new_molData[data][batch][molecule][context]
#     # make sure that molecules in each minibatch has the same size
#     for minibatch in new_molData:
#         minibatchData = minibatch[0] # gets the data only. minibatch[1] contains the classes
#         mask_of_minibatch = []
#         largest_size = 0
#         for mol in minibatchData:
#             if len(mol) > largest_size:
#                 largest_size = len(mol)

#         # adds fake molecules 
#         for mol in minibatchData:
#             mask_of_minibatch.append([0]* len(mol) + [-999]*(largest_size - len(mol)))
#             mol.extend([fake]*(largest_size - len(mol)))
#         minibatch.append(mask_of_minibatch)
        
#     return new_molData

# def load_restrictions(protein_name):
#     with open('../data/protein.groups', 'r') as f:
#         groups = f.readlines()
#     groups = [x.strip('\n').split(',') for x in groups]
#     groups = [x for x in groups if protein_name in x][0]
    
#     with open('../data/protein.cross_enrichment', 'r') as f:
#         cross_enrichment = f.readlines()
#     cross_enrichment = [x.strip('\n').split(',') for x in cross_enrichment]
#     cross_enrichment = [x for x in cross_enrichment if x[0] == protein_name]
#     if cross_enrichment:
#         groups.extend(cross_enrichment[0])
#     return set(groups)


# In[ ]:




