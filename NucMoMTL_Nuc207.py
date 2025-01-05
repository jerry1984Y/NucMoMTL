
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

    train_y = []
    task_ids = []

    for data in batch_traindata:
        feature_plms.append(data[0])

        train_y.append(data[1])
        task_ids.append(data[2])
    data_length = [len(data) for data in feature_plms]

    feature_plms = torch.nn.utils.rnn.pad_sequence(feature_plms, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    task_ids = torch.nn.utils.rnn.pad_sequence(task_ids, batch_first=True, padding_value=0)
    return feature_plms, train_y, torch.tensor(data_length), task_ids

class BioinformaticsDataset(Dataset):
    # X: list of filename
    def __init__(self, X,imemorylist,tasks):
        self.X = X
        self.ilist = imemorylist
        self.Tasks = tasks
        if len(self.ilist)==0:
            for filename in X:
                # esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
                df0 = pd.read_csv('DataSet/prot_embedding/' + filename + '.data', header=None)  # .data  .pssm
                prot = df0.values.astype(float).tolist()

                prot = torch.tensor(prot)

                df2 = pd.read_csv('DataSet/prot_embedding/' + filename + '.label', header=None)
                label = df2.values.astype(int).tolist()
                label = torch.tensor(label)
                # reduce 2D to 1D
                label = torch.squeeze(label)
                taskid = 0
                find = False
                for taskname in self.Tasks:
                    if '_' + taskname in filename:
                        find = True
                        break
                    taskid += 1
                if not find:
                    taskid = 0
                task_id_label = torch.ones(prot.shape[0], dtype=int) * taskid
                tmp=[]
                tmp.append(prot)
                tmp.append(label)
                tmp.append(task_id_label)
                self.ilist.append(tmp)
    def __getitem__(self, index):
        return self.ilist[index][0], self.ilist[index][1],self.ilist[index][2]
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

def train(modelstoreapl):
    train_set = BioinformaticsDataset(train_file_list,memorytrainlist,global_task)
    model = MTLModule()
    epochs = 31

    model = model.to(device)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True,
                              collate_fn=coll_paddding)
    best_val_loss = 3000

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    per_cls_weights = torch.FloatTensor([0.15, 0.85]).to(device)
    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2)
    model.train()
    for j in range(epochs):

        epoch_loss_train = 0.0
        nb_train = 0
        for prot_xs, data_ys, lengths, taskids in train_loader:
            task_outs, diff_losses = model(prot_xs.to(device))
            taskids = taskids.to(device)
            # lengths = lengths.to('cpu')
            data_ys = data_ys.to(device)
            for i in range(len(global_task)):
                task_outs[i] = torch.nn.utils.rnn.pack_padded_sequence(task_outs[i], lengths, batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)

            taskids = torch.nn.utils.rnn.pack_padded_sequence(taskids, lengths, batch_first=True)

            # batchsize=indexs.size(0)
            loss_task = 0
            for i in range(len(global_task)):
                indexs = torch.nonzero(taskids.data == i).squeeze()
                pred = task_outs[i].data[indexs]
                lbs = data_ys.data[indexs]
                if lbs.shape[0] > 0:
                    fc = fcloss(pred, lbs)
                    loss_task += fc
            # diff_losses*0.001 for pssm
            # diff_losses*0.01 for T5,EMS-2
            loss_task = loss_task + diff_losses * 0.01
            optimizer.zero_grad()
            loss_task.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + loss_task.item()
            nb_train += 1
        epoch_loss_avg = epoch_loss_train / nb_train
        print(j, " epoch_loss_avg: ", epoch_loss_avg)
        if best_val_loss > epoch_loss_avg:
            model_fn = modelstoreapl
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg

def test(modelstoreapl):
    test_set = BioinformaticsDataset(test_file_list, memorytestlist, global_task)

    test_load = DataLoader(dataset=test_set, batch_size=32, collate_fn=coll_paddding)
    model = MTLModule()
    model = model.to(device)

    print("==========================Test RESULT================================")

    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()
    tmresult = {}
    # nucs = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP']
    predicted_probs = [[] for i in range(len(global_task))]
    labels_actual = [[] for i in range(len(global_task))]
    labels_predicted = [[] for i in range(len(global_task))]

    with torch.no_grad():
        for prot_xs, data_ys, lengths, taskids in test_load:
            task_outs, diff_losses = model(prot_xs.to(device))
            for i in range(len(global_task)):
                task_outs[i] = torch.nn.utils.rnn.pack_padded_sequence(task_outs[i], lengths.to('cpu'),
                                                                       batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)
            taskids = torch.nn.utils.rnn.pack_padded_sequence(taskids, lengths, batch_first=True)

            for i in range(len(global_task)):
                indexs = torch.nonzero(taskids.data == i).squeeze()
                task_pred = task_outs[i].data[indexs]
                lbs = data_ys.data[indexs]
                task_pred = torch.nn.functional.softmax(task_pred, dim=1)
                task_pred = task_pred.to('cpu')
                if lbs.shape[0] > 0:
                    predicted_probs[i].extend(task_pred[:, 1])
                    labels_actual[i].extend(lbs)
                    labels_predicted[i].extend(torch.argmax(task_pred, dim=1))

        itask_names = global_task
        itaskid = [i for i in range(len(itask_names))]
        for id, task_name in zip(itaskid, itask_names):
            sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult(task_name, labels_actual[id],
                                                                                     predicted_probs[id],
                                                                                     labels_predicted[id])
            tmresult[task_name] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
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
    memorytrainlist = []
    memorytestlist = []

    trainfiles = [  'DataSet/Train/Nuc_207_train_all.txt'
        #'DataSet/Train/pretrain_train_all227.txt',
        #'DataSet/Train/pretrain_train_all221.txt'
    ]
    testfiles = [ 'DataSet/Test/Nuc_207_test_all.txt'
        #'DataSet/Test/pretrain_test_all_17.txt',
        #'DataSet/Test/pretrain_test_all_50.txt'
    ]
    pls = ['NucMoMTL207'
    ]
    global_task = ['TMP', 'CTP', 'CMP', 'UTP', 'UMP']
    circle = 5

    for trains, tests, pl in zip(trainfiles, testfiles, pls):
        train_file_list = read_data_file_trip(trains)
        test_file_list = read_data_file_trip(tests)
        totalkv = {task: [] for task in global_task}
        memorytrainlist = []
        memorytestlist = []
        for i in range(circle):
            storeapl = 'AddTest/MTLADD_MTL-207—_7_part_oc_T5_all_' + pl + '_' + str(i) + '.pkl'
            train(storeapl)
            tmresult = test(storeapl)

            for task in global_task:
                totalkv[task].append(tmresult[task])
            torch.cuda.empty_cache()

        with open('AddTest/Result_MTLT-207_7_part_OC_T5_MTL_' + pl + '.txt', 'w') as f:
            for nuc in global_task:
                np.savetxt(f, totalkv[nuc], delimiter=',', footer='Above is  record ' + nuc, fmt='%s')
                m = np.mean(totalkv[nuc], axis=0)
                np.savetxt(f, [m], delimiter=',', footer='----------Above is AVG -------' + nuc, fmt='%s')

