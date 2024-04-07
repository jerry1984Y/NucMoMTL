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

def read_file_list_from_fasta(filename):
    f = open(filename)
    data = f.readlines()
    f.close()
    results=[]
    block=len(data)//2
    for index in range(block):
        item=data[index*2+0].strip()
        name =item.replace('>','').strip()
        results.append(name)
    return results

def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]
    block=len(data)//2
    for index in range(block):
        item=data[index*2+0].split()
        name =item[0].strip()
        results.append(name)
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
        #esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        df0 = pd.read_csv('customer_test/' + filename + '.data', header=None)
        prot = df0.values.astype(float).tolist()

        prot = torch.tensor(prot)


        df2= pd.read_csv('customer_test/'+  filename+'.label', header=None)
        label = df2.values.astype(int).tolist()
        label = torch.tensor(label)
        #reduce 2D to 1D
        label=torch.squeeze(label)

        return prot, label,index


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
    def diff_loss(self,shared_embeding1,shared_embeding2):
        innermul=shared_embeding1*shared_embeding2
        l2=torch.norm(innermul,dim=2)
        return torch.sum(l2)

    def forward(self,prot_input,datalengths):
        prot_input=prot_input.permute(0,2,1)
        prot1=self.share_task_block1(prot_input)
        prot2=self.share_task_block2(prot_input)
        prot3=self.share_task_block3(prot_input)
        prot1 = prot1.permute(0,2,1)
        prot2 = prot2.permute(0, 2, 1)
        prot3 = prot3.permute(0, 2, 1)

        diff_loss1 = self.diff_loss(prot1, prot2)
        diff_loss2 = self.diff_loss(prot2, prot3)
        diff_loss3 = self.diff_loss(prot1, prot3)
        diff_losses = diff_loss1 + diff_loss2 + diff_loss3

        protd=prot1+ prot2+prot3

        nuclist=[]
        for i in range(5):
            nuclist.append(self.nuc_along_share_tasks_fc[i](protd))
        return nuclist,diff_losses


def test(modelstoreapl,intputfile):
    test_file_list = read_data_file_trip(intputfile)
    test_set = BioinformaticsDataset(test_file_list)

    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)
    model = MTLModule()
    model = model.to(device)

    print("==========================Test RESULT================================")

    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()

    predicted_probs_adp = []
    labels_actual_adp = []
    labels_predicted_adp = []

    predicted_probs_amp = []
    labels_actual_amp = []
    labels_predicted_amp = []

    predicted_probs_atp = []
    labels_actual_atp = []
    labels_predicted_atp = []

    predicted_probs_gdp = []
    labels_actual_gdp = []
    labels_predicted_gdp = []

    predicted_probs_gtp = []
    labels_actual_gtp = []
    labels_predicted_gtp = []

    with torch.no_grad():
        for prot_xs,data_ys, lengths,indexs in  test_load:
            nuclist,diff_losses= model(prot_xs.to(device), lengths.to(device))
            for nuc_adp,nuc_amp,nuc_gdp,nuc_gtp,nuc_atp,data_y,index,length in zip(nuclist[0],nuclist[1],nuclist[2],
                                                    nuclist[3],nuclist[4],data_ys,indexs,lengths):
                index=index.numpy()
                length=length.numpy()
                data_y=data_y[0:length]
                adp_pred = torch.nn.functional.softmax(nuc_adp[0:length], dim=1)
                adp_pred=adp_pred.to('cpu')

                amp_pred = torch.nn.functional.softmax(nuc_amp[0:length], dim=1)
                amp_pred = amp_pred.to('cpu')
                gdp_pred = torch.nn.functional.softmax(nuc_gdp[0:length], dim=1)
                gdp_pred = gdp_pred.to('cpu')

                gtp_pred = torch.nn.functional.softmax(nuc_gtp[0:length], dim=1)
                gtp_pred = gtp_pred.to('cpu')

                atp_pred = torch.nn.functional.softmax(nuc_atp[0:length], dim=1)
                atp_pred = atp_pred.to('cpu')

                #filename=test_file_list[index]

                #if '_ADP' in filename:
                predicted_probs_adp.extend(adp_pred[:, 1])
                labels_actual_adp.extend(data_y)
                labels_predicted_adp.extend(torch.argmax(adp_pred, dim=1))
                #elif '_AMP' in filename:
                predicted_probs_amp.extend(amp_pred[:, 1])
                labels_actual_amp.extend(data_y)
                labels_predicted_amp.extend(torch.argmax(amp_pred, dim=1))
                #elif '_GDP' in filename:
                predicted_probs_gdp.extend(gdp_pred[:, 1])
                labels_actual_gdp.extend(data_y)
                labels_predicted_gdp.extend(torch.argmax(gdp_pred, dim=1))
                #elif '_GTP' in filename:
                predicted_probs_gtp.extend(gtp_pred[:, 1])
                labels_actual_gtp.extend(data_y)
                labels_predicted_gtp.extend(torch.argmax(gtp_pred, dim=1))
                #else:  #ATP
                predicted_probs_atp.extend(atp_pred[:, 1])
                labels_actual_atp.extend(data_y)
                labels_predicted_atp.extend(torch.argmax(atp_pred, dim=1))
    #con


    print('-------------->')
    printresult('ATP', labels_actual_atp, predicted_probs_atp, labels_predicted_atp)
    printresult('ADP', labels_actual_adp, predicted_probs_adp, labels_predicted_adp)
    printresult('AMP', labels_actual_amp, predicted_probs_amp, labels_predicted_amp)
    printresult('GTP', labels_actual_gtp, predicted_probs_gtp, labels_predicted_gtp)
    printresult('GDP', labels_actual_gdp,predicted_probs_gdp, labels_predicted_gdp)




    #save_prob_label(predicted_probs_amp, labels_actual_amp, 'CaseStudy_NucMoMTL_AMP')
    print('<----------------save to csv finish')

def save_prob_label(probs,labels,filename):
    probs = np.array(probs)
    labels = np.array(labels)
    data = np.hstack((probs.reshape(-1, 1), labels.reshape(-1, 1)))
    names = ['probs', 'actuallabels']
    Pd_data = pd.DataFrame(columns=names, data=data)
    Pd_data.to_csv(filename)

def printresult(ligand,actual_label,predict_prob,predict_label):
    print('\n---------',ligand,'-------------')
    auc = metrics.roc_auc_score(actual_label, predict_prob)
    precision_1, recall_1, threshold_1 = metrics.precision_recall_curve(actual_label, predict_prob)
    aupr_1 = metrics.auc(recall_1, precision_1)
    acc=metrics.accuracy_score(actual_label, predict_label)
    print('acc ',acc )
    print('balanced_accuracy ', metrics.balanced_accuracy_score(actual_label, predict_label))
    tn, fp, fn, tp = metrics.confusion_matrix(actual_label, predict_label).ravel()
    print('tn, fp, fn, tp ', tn, fp, fn, tp)
    mcc=metrics.matthews_corrcoef(actual_label, predict_label)
    print('MCC ', mcc)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = 2 * tp / (2 * tp + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    youden = sensitivity + specificity - 1
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('AUPR ', aupr_1)
    save_prob_label(predict_prob, actual_label, 'customer_test/'+ligand+'.csv')
    print('---------------END------------')

def extratdata(file,destfolder):
    student_tuples=read_data_file_trip(file)
    i=1
    recordlength=len(student_tuples)
    for name, seq, length, label in student_tuples:
        print(f'progress {i}/{recordlength}')
        i += 1
        with open(os.path.join(destfolder, name + '.label'), 'w') as f:
            f.write(','.join(l for l in label))
        newseq=' '.join(s for s in seq)
        newseq=re.sub(r"[UZOB]", "X", newseq)
        #print('newseq length',len(newseq))
        ids = tokenizer.batch_encode_plus([newseq], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            #features.append(seq_emd)
            with open(os.path.join(destfolder, name + '.data'), 'w') as f:
                np.savetxt(f, seq_emd, delimiter=',', fmt='%s')

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    #extract embedding
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model = model.eval()
    testfilename='customer_test/query.txt'
    extratdata(testfilename, 'customer_test/')


    test('pre_model/S2MTLTT5Msl2ADDMTL_S2Nuc-CNN_M_scale_l2ADD_02__T5_1.pkl',testfilename)


