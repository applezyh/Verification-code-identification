# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.nn as nn
import torch
from RNNnew import RNN
from set import dataset, lable_set
import cv2 as cv
lr = 0.001
batch_size = 100
batch = 0
net = torch.load("net.pkl")
net.cuda()
opt = torch.optim.Adam(net.parameters(), lr=lr)
sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.5)
data = dataset()
lable = lable_set().cuda()
loss_func = nn.CrossEntropyLoss()
for i in range(batch):
    for step in range(75):
        in_data = data[step*batch_size:step*batch_size+batch_size]
        result = net(in_data)
        re1 = result[0]
        re2 = result[1]
        re3 = result[2]
        re4 = result[3]
        re = lable[step*batch_size:step*batch_size+batch_size]
        loss1 = loss_func(re1, re[:, :1].reshape(batch_size))
        loss2 = loss_func(re2, re[:, 1:2].reshape(batch_size))
        loss3 = loss_func(re3, re[:, 2:3].reshape(batch_size))
        loss4 = loss_func(re4, re[:, 3:].reshape(batch_size))
        opt.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward(retain_graph=True)
        loss4.backward(retain_graph=True)
        opt.step()
    if i % 5 == 0:
        sch.step()
        print(loss1 + loss2 + loss3 + loss4)
        torch.save(net, "net.pkl")

test = data[0:10000]
count = 0
for i in range(1):
    r0 = net(test[i*10:i*10+10].cuda())
    for j in range(10):
        r1 = torch.argmax(r0[0][j]).detach().cpu().numpy()
        r2 = torch.argmax(r0[1][j]).detach().cpu().numpy()
        r3 = torch.argmax(r0[2][j]).detach().cpu().numpy()
        r4 = torch.argmax(r0[3][j]).detach().cpu().numpy()
        print(chr(r1+97), chr(r2+97), chr(r3+97), chr(r4+97))
        img = data[j+i*10].reshape(60, 160, 1).detach().cpu().numpy()
        cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.imshow("img", img)
        cv.waitKey(0)
        if r1 == lable[i*10+j][0] and r2 == lable[i*10+j][1] and r3 == lable[i*10+j][2] and r4 == lable[i*10+j][3]:
            count = count + 1