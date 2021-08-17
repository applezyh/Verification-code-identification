import torch
import torch.nn as nn
from resNet import ResNet18

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.resnet = ResNet18()
        self.elu = nn.ELU()
        self.ln2 = nn.Linear(1024, 512)
        self.ln3 = nn.Linear(512, 256)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=5,
            batch_first=True,
        )
        self.bn1d = nn.BatchNorm1d(256)
        self.ou11 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
        )
        self.ou12 = nn.Sequential(
            nn.Linear(256, 26),
            nn.Sigmoid()
        )
        self.ou21 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),

        )
        self.ou22 = nn.Sequential(
            nn.Linear(256, 26),
            nn.Sigmoid()
        )
        self.ou31 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
        )
        self.ou32 = nn.Sequential(
            nn.Linear(256, 26),
            nn.Sigmoid()
        )
        self.ou41 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
        )
        self.ou42 = nn.Sequential(
            nn.Linear(256, 26),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.ln2(self.elu(x))
        x3 = self.ln3(self.elu(x))
        x = torch.cat((x3, x3.clone(), x3.clone(), x3.clone()), dim=1)
        x = x.reshape(x.shape[0], 4, 256)
        x = self.rnn(x)[0]
        out1 = x[:, :1, :].reshape(x.shape[0], 512)
        out2 = x[:, 1:2, :].reshape(x.shape[0], 512)
        out3 = x[:, 2:3, :].reshape(x.shape[0], 512)
        out4 = x[:, 3:, :].reshape(x.shape[0], 512)
        out1 = self.bn1d(self.ou11(out1)+x3)
        out2 = self.bn1d(self.ou21(out2)+x3)
        out3 = self.bn1d(self.ou31(out3)+x3)
        out4 = self.bn1d(self.ou41(out4)+x3)
        out1 = self.ou12(out1)
        out2 = self.ou22(out2)
        out3 = self.ou32(out3)
        out4 = self.ou42(out4)
        return out1, out2, out3, out4