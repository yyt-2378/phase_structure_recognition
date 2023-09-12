import time
import os
from model import UNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset_atom import StemDataset
from torch.utils.tensorboard import SummaryWriter


net = UNet().cuda()
optimizer = torch.optim.Adam(net.parameters(),  lr=0.0001)
loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0])).cuda()
data_dir = 'F:\\STEM_new_VAE_Data'
train_dataset = StemDataset(mode='train', dir=data_dir)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
print('Dataset loaded! length of train set is {0}'.format(len(train_dataset)))
summary = SummaryWriter(r'Log')
EPOCH = 200
print('load net')
if os.path.exists('SAVE/Unet.pt'):
    print("The file 'Unet.pt' exists.")
    net.load_state_dict(torch.load('SAVE/Unet.pt'))
    print('load success')
else:
    print("The file 'Unet.pt' does not exist.")

start_time = time.time()  # 开始训练的时间
for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    net.train()
    for i, (img, label) in enumerate(dataloader):
        img = img.cuda()
        label = label.cuda()
        img_out = net(img)
        loss = loss_func(img_out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_total = label.size(0)

        # print train loss
        end_time = time.time()  # 训练结束时间
        print("训练时间: {}".format(end_time - start_time))
        print("训练次数: {}, Loss: {}".format(i, loss))
        summary.add_scalar("train_loss", loss, i)

    torch.save(net.state_dict(), r'SAVE/Unet.pt')
    summary.close()
    print('第{}轮结束'.format(epoch))
