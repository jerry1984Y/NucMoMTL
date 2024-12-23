
import numpy as np
import pandas as pd
import torch
import  torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LossFunction.focalLoss import FocalLoss_v2


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
    #f0agv=[]
    #feature_evo = []

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
        df0 = pd.read_csv('DataSet/prot_embedding/' + filename + '.data', header=None)
        prot = df0.values.astype(float).tolist()

        prot = torch.tensor(prot)


        df2= pd.read_csv('DataSet/prot_embedding/'+  filename+'.label', header=None)
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
        for i in range(10):
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
        for i in range(10):
            nuclist.append(self.nuc_along_share_tasks_fc[i](protd))
        return nuclist,diff_losses

def train(modelstoreapl):
    train_set = BioinformaticsDataset(train_file_list)
    model = MTLModule()
    epochs = 40

    model = model.to(device)
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
                              persistent_workers=True,
                              collate_fn=coll_paddding)
    best_val_loss = 3000

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    per_cls_weights = torch.FloatTensor([0.15, 0.85]).to(device)
    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2)
    model.train()
    for i in range(epochs):

        epoch_loss_train = 0.0
        nb_train = 0
        for prot_xs, data_ys, lengths,indexs in train_loader:
            optimizer.zero_grad()
            ps=prot_xs.to(device)

            nuclist,diff_losses = model(ps, lengths.to(device))
            # nucs = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP',adp,amp,atp,gdp,gtp]
            tmp_loss=0
            ctp_loss=0
            cmp_loss=0
            utp_loss=0
            ump_loss=0
            adp_loss = 0
            amp_loss = 0
            atp_loss = 0
            gdp_loss = 0
            gtp_loss = 0
            total_length=0
            #batchsize=indexs.size(0)
            for nuc_tmp,nuc_ctp,nuc_cmp,nuc_utp,nuc_ump,nuc_adp,nuc_amp,nuc_gdp,nuc_gtp,nuc_atp,data_y,index,length in zip(nuclist[0],nuclist[1],nuclist[2],
                                                    nuclist[3],nuclist[4],nuclist[5],nuclist[6],nuclist[7],nuclist[8],
                                                    nuclist[9],data_ys,indexs,lengths):
                index=index.numpy()
                length=length.numpy()
                data_y=data_y.to(device)
                total_length+=length
                filename=train_file_list[index]
                if '_TMP' in filename:
                    tmp_loss+=fcloss(nuc_tmp[0:length], data_y[0:length])
                elif '_CTP' in filename:
                    ctp_loss += fcloss(nuc_ctp[0:length], data_y[0:length])
                elif '_CMP' in filename:
                    cmp_loss += fcloss(nuc_cmp[0:length], data_y[0:length])
                elif '_UTP' in filename:
                    utp_loss += fcloss(nuc_utp[0:length], data_y[0:length])
                elif '_UMP' in filename:
                    ump_loss += fcloss(nuc_ump[0:length], data_y[0:length])
                elif '_ADP' in filename:
                    adp_loss+=fcloss(nuc_adp[0:length], data_y[0:length])
                elif '_AMP' in filename:
                    amp_loss += fcloss(nuc_amp[0:length], data_y[0:length])
                elif '_GDP' in filename:
                    gdp_loss += fcloss(nuc_gdp[0:length], data_y[0:length])
                elif '_GTP' in filename:
                    gtp_loss += fcloss(nuc_gtp[0:length], data_y[0:length])
                else:  #ATP
                    atp_loss += fcloss(nuc_atp[0:length], data_y[0:length])
            #diff_losses*0.001 for pssm
            #diff_losses*0.01 for T5,EMS-2
            lose=tmp_loss+ctp_loss+cmp_loss+utp_loss+ump_loss+adp_loss+amp_loss+atp_loss+gdp_loss+gtp_loss+diff_losses*0.01

            lose.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + lose.item()
            nb_train += 1
        epoch_loss_avg = epoch_loss_train / nb_train
        if best_val_loss > epoch_loss_avg:
            model_fn = modelstoreapl
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            if i % 10 == 0:
                print('epochs ', i)
                print("Save model, best_val_loss: ", best_val_loss)

def test(modelstoreapl):
    test_set = BioinformaticsDataset(test_file_list)

    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)
    model = MTLModule()
    model = model.to(device)

    print("==========================Test RESULT================================")

    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()
    # nucs = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP']
    predicted_probs_tmp = []
    labels_actual_tmp = []
    labels_predicted_tmp = []

    predicted_probs_ctp = []
    labels_actual_ctp = []
    labels_predicted_ctp = []

    predicted_probs_cmp = []
    labels_actual_cmp = []
    labels_predicted_cmp = []

    predicted_probs_utp = []
    labels_actual_utp = []
    labels_predicted_utp = []

    predicted_probs_ump = []
    labels_actual_ump = []
    labels_predicted_ump = []

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
    # nucs = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP']
    with torch.no_grad():
        for prot_xs,data_ys, lengths,indexs in  test_load:
            nuclist,diff_losses= model(prot_xs.to(device), lengths.to(device))
            for nuc_tmp,nuc_ctp,nuc_cmp,nuc_utp,nuc_ump,nuc_adp,nuc_amp,nuc_gdp,nuc_gtp,nuc_atp,data_y,index,length in zip(nuclist[0],nuclist[1],nuclist[2],
                                                    nuclist[3],nuclist[4],nuclist[5],nuclist[6],nuclist[7],nuclist[8],
                                                    nuclist[9],data_ys,indexs,lengths):
                index=index.numpy()
                length=length.numpy()
                data_y=data_y[0:length]
                tmp_pred = torch.nn.functional.softmax(nuc_tmp[0:length], dim=1)
                tmp_pred=tmp_pred.to('cpu')

                ctp_pred = torch.nn.functional.softmax(nuc_ctp[0:length], dim=1)
                ctp_pred = ctp_pred.to('cpu')

                cmp_pred = torch.nn.functional.softmax(nuc_cmp[0:length], dim=1)
                cmp_pred = cmp_pred.to('cpu')

                utp_pred = torch.nn.functional.softmax(nuc_utp[0:length], dim=1)
                utp_pred = utp_pred.to('cpu')

                ump_pred = torch.nn.functional.softmax(nuc_ump[0:length], dim=1)
                ump_pred = ump_pred.to('cpu')

                adp_pred = torch.nn.functional.softmax(nuc_adp[0:length], dim=1)
                adp_pred = adp_pred.to('cpu')

                amp_pred = torch.nn.functional.softmax(nuc_amp[0:length], dim=1)
                amp_pred = amp_pred.to('cpu')
                gdp_pred = torch.nn.functional.softmax(nuc_gdp[0:length], dim=1)
                gdp_pred = gdp_pred.to('cpu')

                gtp_pred = torch.nn.functional.softmax(nuc_gtp[0:length], dim=1)
                gtp_pred = gtp_pred.to('cpu')

                atp_pred = torch.nn.functional.softmax(nuc_atp[0:length], dim=1)
                atp_pred = atp_pred.to('cpu')

                filename=test_file_list[index]
                #nucs = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP']
                if '_TMP' in filename:
                    predicted_probs_tmp.extend(tmp_pred[:, 1])
                    labels_actual_tmp.extend(data_y)
                    labels_predicted_tmp.extend(torch.argmax(tmp_pred, dim=1))
                elif '_CTP' in filename:
                    predicted_probs_ctp.extend(ctp_pred[:, 1])
                    labels_actual_ctp.extend(data_y)
                    labels_predicted_ctp.extend(torch.argmax(ctp_pred, dim=1))
                elif '_CMP' in filename:
                    predicted_probs_cmp.extend(cmp_pred[:, 1])
                    labels_actual_cmp.extend(data_y)
                    labels_predicted_cmp.extend(torch.argmax(cmp_pred, dim=1))
                elif '_UTP' in filename:
                    predicted_probs_utp.extend(utp_pred[:, 1])
                    labels_actual_utp.extend(data_y)
                    labels_predicted_utp.extend(torch.argmax(utp_pred, dim=1))
                elif '_UMP' in filename:
                    predicted_probs_ump.extend(ump_pred[:, 1])
                    labels_actual_ump.extend(data_y)
                    labels_predicted_ump.extend(torch.argmax(ump_pred, dim=1))
                elif '_ADP' in filename:
                    predicted_probs_adp.extend(adp_pred[:, 1])
                    labels_actual_adp.extend(data_y)
                    labels_predicted_adp.extend(torch.argmax(adp_pred, dim=1))
                elif '_AMP' in filename:
                    predicted_probs_amp.extend(amp_pred[:, 1])
                    labels_actual_amp.extend(data_y)
                    labels_predicted_amp.extend(torch.argmax(amp_pred, dim=1))
                elif '_GDP' in filename:
                    predicted_probs_gdp.extend(gdp_pred[:, 1])
                    labels_actual_gdp.extend(data_y)
                    labels_predicted_gdp.extend(torch.argmax(gdp_pred, dim=1))
                elif '_GTP' in filename:
                    predicted_probs_gtp.extend(gtp_pred[:, 1])
                    labels_actual_gtp.extend(data_y)
                    labels_predicted_gtp.extend(torch.argmax(gtp_pred, dim=1))
                else:  #ATP
                    predicted_probs_atp.extend(atp_pred[:, 1])
                    labels_actual_atp.extend(data_y)
                    labels_predicted_atp.extend(torch.argmax(atp_pred, dim=1))
    #con


    print('-------------->')
    tmresult = {}
    # nucs = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP']
    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('TMP', labels_actual_tmp,
                                                                             predicted_probs_tmp, labels_predicted_tmp)
    tmresult['TMP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]

    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('CTP', labels_actual_ctp,
                                                                             predicted_probs_ctp, labels_predicted_ctp)
    tmresult['CTP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('CMP', labels_actual_cmp,
                                                                             predicted_probs_cmp, labels_predicted_cmp)
    tmresult['CMP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('UTP', labels_actual_utp,
                                                                             predicted_probs_utp, labels_predicted_utp)
    tmresult['UTP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('UMP', labels_actual_ump,
                                                                             predicted_probs_ump, labels_predicted_ump)
    tmresult['UMP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]

    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('ADP', labels_actual_adp,
                                                                             predicted_probs_adp, labels_predicted_adp)
    tmresult['ADP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]

    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('AMP', labels_actual_amp,
                                                                             predicted_probs_amp, labels_predicted_amp)
    tmresult['AMP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('ATP', labels_actual_atp,
                                                                             predicted_probs_atp, labels_predicted_atp)
    tmresult['ATP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('GDP', labels_actual_gdp,
                                                                             predicted_probs_gdp, labels_predicted_gdp)
    tmresult['GDP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('GTP', labels_actual_gtp,
                                                                             predicted_probs_gtp, labels_predicted_gtp)
    tmresult['GTP'] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]

    return tmresult


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
    print('---------------END------------')
    return sensitivity, specificity, acc, precision, mcc, auc, aupr_1
if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")


    trainfiles = [  'DataSet/Train/Nuc_207_1521_Train_all.txt' ]
    testfiles = [ 'DataSet/Test/Nuc_207_1521_Test_all.txt']
    pls = ['NucMoMTL207_1521'
    ]

    circle = 5

    for trains, tests, pl in zip(trainfiles, testfiles, pls):
        train_file_list = read_data_file_trip(trains)
        test_file_list = read_data_file_trip(tests)
        totalkv = {'TMP': [], 'CTP': [], 'CMP': [], 'UTP': [], 'UMP': [],'ATP': [], 'ADP': [], 'AMP': [], 'GDP': [], 'GTP': []}
        for i in range(circle):
            storeapl = 'T5207_1521MTLADD_l2_MTL_' + pl + '_' + str(i) + '.pkl'
            train(storeapl)
            tmresult = test(storeapl)

            totalkv['TMP'].append(tmresult['TMP'])
            totalkv['CTP'].append(tmresult['CTP'])
            totalkv['CMP'].append(tmresult['CMP'])
            totalkv['UTP'].append(tmresult['UTP'])
            totalkv['UMP'].append(tmresult['UMP'])
            totalkv['ATP'].append(tmresult['ATP'])
            totalkv['ADP'].append(tmresult['ADP'])
            totalkv['AMP'].append(tmresult['AMP'])
            totalkv['GDP'].append(tmresult['GDP'])
            totalkv['GTP'].append(tmresult['GTP'])
            torch.cuda.empty_cache()

        with open('T5207_1521MTLADD_l2_MTL_' + pl + '.txt', 'w') as f:
            nucs = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP','ATP', 'ADP', 'AMP', 'GDP', 'GTP']
            for nuc in nucs:
                np.savetxt(f, totalkv[nuc], delimiter=',', footer='Above is  record ' + nuc, fmt='%s')
                m = np.mean(totalkv[nuc], axis=0)
                np.savetxt(f, [m], delimiter=',', footer='----------Above is AVG -------' + nuc, fmt='%s')


