import os
import re

import numpy as np
import pandas as pd
import torch
import  torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LossFunction.focalLoss import FocalLoss_v2
from transformers import T5EncoderModel, T5Tokenizer


def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]

    results.append(data[0].strip())
    return results

def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature_plms = []


    train_y = []
    indexs=[]

    for data in batch_traindata:
        feature_plms.append(data[0])
        #f0agv.append(data[1])
        #feature_evo.append(data[1])
        train_y.append(data[1])
        indexs.append(data[2])
    data_length = [len(data) for data in feature_plms]
    #
    # mask = torch.full((len(batch_traindata), data_length[0]), False).bool()  # crete init mask
    # for mi, aci in zip(mask, data_length):
    #     mi[aci:] = True

    feature_plms = torch.nn.utils.rnn.pad_sequence(feature_plms, batch_first=True, padding_value=0)
    #f0agv = torch.nn.utils.rnn.pad_sequence(f0agv, batch_first=True, padding_value=0)
    #feature_evo = torch.nn.utils.rnn.pad_sequence(feature_evo, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    return feature_plms,train_y,torch.tensor(data_length),torch.tensor(indexs)

class BioinformaticsDataset(Dataset):
    # X: list of filename
    def __init__(self, X):
        self.X = X
    def __getitem__(self, index):
        filename = self.X[index]
        print(filename)
        #esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        df0 = pd.read_csv('customer_test/' + filename, header=None)
        prot = df0.values.astype(float).tolist()

        prot = torch.tensor(prot)


        return prot


    def __len__(self):
        return len(self.X)


class MTLModule(nn.Module):
    def __init__(self):
        super(MTLModule,self).__init__()

        self.share_task_block1=nn.Sequential(nn.Conv1d(1024,512,1,padding='same'),
                                            nn.ReLU(True),
                                            nn.Conv1d(512,256,1,padding='same'),
                                            nn.ReLU(True),
                                            nn.Conv1d(256,128,1,padding='same'),
                                            nn.ReLU(True))
        self.share_task_block2 = nn.Sequential(nn.Conv1d(1024, 512, 3, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 3, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 3, padding='same'),
                                               nn.ReLU(True))
        self.share_task_block3 = nn.Sequential(nn.Conv1d(1024, 512, 5, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 5, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 5, padding='same'),
                                               nn.ReLU(True))

        self.nuc_along_share_tasks_fc=nn.ModuleList()
        for i in range(5):
            self.nuc_along_share_tasks_fc.append(nn.Sequential(nn.Linear(128,512),
                                          nn.Dropout(0.5),
                                          nn.Linear(512,64),
                                          nn.Dropout(0.5),
                                          nn.Linear(64,2)))

    def update_ol(self):
        reg = 1e-6
        orth_loss = torch.zeros(1).to(device)
        #need consider all in OC or only shared encoder in OC
        #all in oc Nuc-798,849
        #part in oc Nuc-1521
        for name, param in self.named_parameters():
            if 'bias' in name or 'nuc_along_share_tasks_fc' in name: #  or 'nuc_along_share_tasks_fc' in name
                continue
            else:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0]).to(device)
                orth_loss = orth_loss + (reg * sym.abs().sum())
        return orth_loss
    def forward(self,prot_input):
        prot_input=prot_input.permute(0,2,1)
        prot1=self.share_task_block1(prot_input)
        prot2=self.share_task_block2(prot_input)
        prot3=self.share_task_block3(prot_input)
        prot1 = prot1.permute(0,2,1)
        prot2 = prot2.permute(0, 2, 1)
        prot3 = prot3.permute(0, 2, 1)

        protd=prot1+ prot2+prot3

        nuclist=[]
        for i in range(5):
            nuclist.append(self.nuc_along_share_tasks_fc[i](protd))
        return nuclist,self.update_ol()

def test(modelstoreapl,intputfile):
    test_set = BioinformaticsDataset([intputfile])

    test_load = DataLoader(dataset=test_set, batch_size=1,pin_memory=True, persistent_workers=True)
    model = MTLModule()
    model = model.to(device)

    print("==========================Test RESULT================================")

    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()


    task_preds = []
    with torch.no_grad():
        for prot_xs in  test_load:
            nuclist,diff_losses= model(prot_xs.to(device))

            for i in range(len(nuclist)):
                task_pred = nuclist[i]
                task_pred = task_pred.to('cpu')
                task_pred = task_pred[0]
                task_pred = torch.nn.functional.softmax(task_pred, dim=1)
                task_preds.append(np.array(task_pred[:, 1]))
    return task_preds
    #nuc_adp,nuc_amp,nuc_gdp,nuc_gtp,nuc_atp


def extratdata(filects,destfolder):
    seq=filects[0]
    newseq=' '.join(s for s in seq)
    newseq=re.sub(r"[UZOB]", "X", newseq)
    ids = tokenizer.batch_encode_plus([newseq], add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        with open(os.path.join(destfolder, tmp_embedding), 'w') as f:
            np.savetxt(f, seq_emd, delimiter=',', fmt='%s')


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    tmp_embedding='Tmp.data'
    #extract embedding
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model = model.eval()
    testfilename='customer_test/query.txt'
    student_tuples = read_data_file_trip(testfilename)
    extratdata(student_tuples, 'customer_test/')
    seqs = [s for s in student_tuples[0]]
    pdata = {}
    pdata['sequence'] = seqs
    task_preds=test('pre_model/NucMoMTL-1521_0.pkl',tmp_embedding)
    #nuc_adp, nuc_amp, nuc_gdp, nuc_gtp, nuc_atp
    pdata['ADP']=task_preds[0]
    pdata['AMP']=task_preds[1]
    pdata['GDP']=task_preds[2]
    pdata['GTP']=task_preds[3]
    pdata['ATP']=task_preds[4]
    df = pd.DataFrame(pdata)  # create DataFrame and save to xlsx
    df.to_excel('customer_test/result.xlsx', index=False)
