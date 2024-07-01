import torch
import torch.nn as nn
import torch.nn.functional as F
import csv


class SoftTarget(nn.Module):
    def __init__(self, class_counts, venomous_list='venomous_status_list.csv', temp=3):
        super().__init__()

        self.venomous_list = {}
        class_counts = torch.FloatTensor(class_counts)
        with open(venomous_list, mode='r') as infile:
            reader = csv.reader(infile)
            for i, r in enumerate(reader):
                if i == 0:
                    continue
                self.venomous_list[int(r[0])] = int(r[1])
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6
        self.temp = temp
        self.penalties = [0.1, 1, 10, 10, 100]
        self.target_penalties = {}
        for cls in sorted(self.venomous_list.items()):
            tp = []
            for c in sorted(self.venomous_list.items()):
                if c[0] == cls[0]:
                    tp.append(self.penalties[0])
                elif c[1] == 0 and cls[1] == 0:
                    tp.append(self.penalties[1])
                elif c[1] == 1 and cls[1] == 0:
                    tp.append(self.penalties[2])
                elif c[1] == 1 and cls[1] == 1:
                    tp.append(self.penalties[3])
                elif c[1] == 0 and cls[1] == 1:
                    tp.append(self.penalties[4])
                else:
                    print('FATAL - not conditioned state')
            self.target_penalties[cls[0]] = tp

    def forward(self, logits, targets):
        penalized_target = torch.tensor([self.target_penalties[cls] for cls in targets.tolist()])

        penalized_target = penalized_target.to(logits.get_device())

        loss = F.cross_entropy(logits, F.softmax(-self.temp * penalized_target.float().log(), dim=1),
                               reduction='none').sum(-1)
        return loss / len(targets)


class CEVP(nn.Module):
    def __init__(self, class_counts, venomous_list='venomous_status_list.csv', w_penalty=1.0):
        super().__init__()

        self.venomous_list = {}
        class_counts = torch.FloatTensor(class_counts)
        with open(venomous_list, mode='r') as infile:
            reader = csv.reader(infile)
            for i, r in enumerate(reader):
                if i == 0:
                    continue
                self.venomous_list[int(r[0])] = int(r[1])
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6
        self.w_penalty = w_penalty
        self.penalties = [0, 1, 2, 2, 5]
        self.target_penalties = {}
        for cls in sorted(self.venomous_list.items()):
            tp = []
            for c in sorted(self.venomous_list.items()):
                if c[0] == cls[0]:
                    tp.append(self.penalties[0])
                elif c[1] == 0 and cls[1] == 0:
                    tp.append(self.penalties[1])
                elif c[1] == 1 and cls[1] == 0:
                    tp.append(self.penalties[2])
                elif c[1] == 1 and cls[1] == 1:
                    tp.append(self.penalties[3])
                elif c[1] == 0 and cls[1] == 1:
                    tp.append(self.penalties[4])
                else:
                    print('FATAL - not conditioned state')
            self.target_penalties[cls[0]] = tp

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        candidate = logits.argmax(-1)
        for i, target in enumerate(targets.tolist()):
            penalty = self.target_penalties[target]
            p = penalty[candidate[i]]
            ce[i] = ce[i] + p * self.w_penalty

        loss = ce.sum(-1)
        return loss / len(targets)