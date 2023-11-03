
import math
import torch
import numpy as np
from torch.utils.data import Dataset

class KTDataset(Dataset):
    def __init__(self, q, qa, a):
        self.q = q
        self.qa = qa
        self.a = a

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, index):
        return self.q[index], self.qa[index], self.a[index]

class PID_DATA(object):
    def __init__(self, n_pid,  seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_pid = n_pid

    def load_data(self, path):
        f_data = open(path, 'r')
        a_data = []
        pa_data = []
        p_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 3 == 0:
                student_id = lineID//4
            if lineID % 3 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                n_split = 1
                if len(P) > self.seqlen:
                    n_split = math.floor(len(P) / self.seqlen)
                    if len(P) % self.seqlen:
                        n_split = n_split + 1
                else:
                    pass
                for k in range(n_split):
                    p_sequence = []
                    pa_sequence = []
                    a_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(P[i]) > 0:
                            Xindex = int(P[i]) + int(A[i]) * self.n_pid
                            p_sequence.append(int(P[i]))
                            pa_sequence.append(Xindex)
                            a_sequence.append(int(A[i]))
                        else:
                            print(P[i])
                    p_data.append(p_sequence)
                    pa_data.append(pa_sequence)
                    a_data.append(a_sequence)

        f_data.close()
        pa_dataArray = np.zeros((len(pa_data), self.seqlen))
        for j in range(len(pa_data)):
            dat = pa_data[j]
            pa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat

        a_dataArray = -np.ones((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat
        return KTDataset(p_dataArray, pa_dataArray, a_dataArray)

if __name__ == '__main__':
    dat = PID_DATA(n_pid=16891, seqlen=200, separate_char=',')
    ds = dat.load_data("data/assist2009_pid/3_line/assist2009_pid_valid1.csv")