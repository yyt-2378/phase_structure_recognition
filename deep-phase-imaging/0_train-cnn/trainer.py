from torch.utils.data import DataLoader as DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_model import ResidualUNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import time
from tqdm import tqdm as tqdm
import numpy as np
from get_kfold_data import get_kfold


def train(dataset, model_dir, workers, batch_size, lr, nepoch, dr_rate, Lossfuc) ->None:
    '''
    :param dataset_dir: dataset
    :param model_dir: model directory
    :param wokers: threads number
    :param batch_size: batch size
    :param lr: learning rate
    :param nepoch: epochs
    :return: None
    '''
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    print('Dataset loaded! length of train set is {0}'.format(len(dataset)))

    model = ResidualUNet(dr_rate=dr_rate)                       # 实例化一个网络
    if torch.cuda.is_available():
        model = model.cuda()          # 使用gpu训练
    model = nn.DataParallel(model)      # 传入
    model.train()                       # 训练模式

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)       # Adam优化器

    # 添加tensorboard
    writer = SummaryWriter("logs_train")
    start_time = time.time()  # 开始训练的时间

    total_train_step = 0                    # 训练图片数量
    for epoch in tqdm(range(nepoch)):
        train_loss = []
        for img, label in train_loader:
            img, label = Variable(img), Variable(label)           # 将数据放置在PyTorch的Variable节点
            img, label = img.cuda(), label.cuda()
            out = model(img)
            loss = Lossfuc(out, label)      # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
            total_train_step += 1

            # print('Epoch:{0},Frame:{1}, train_loss {2}, train_accuracy {3}'.format(epoch, total_train_step*batch_size, loss/batch_size, train_accuracy))

            if total_train_step % 40 == 0:
                end_time = time.time()  # 训练结束时间
                print("训练时间: {}".format(end_time - start_time))
                print("训练次数: {}, Loss: {}".format(total_train_step, np.mean(train_loss)))
                writer.add_scalar("train_loss", np.mean(train_loss), total_train_step)

        tqdm.write('Epoch {0}, train_loss {1}'.format(epoch, np.mean(train_loss)))

    torch.save(model.state_dict(), '{0}/model.pth'.format(model_dir))  # 训练所有数据后，保存网络的参数
    writer.close()


if __name__ == '__main__':
    dataset_dir = './data/'  # 数据集路径
    model_dir = './model/'  # 网络参数保存位置
    workers = 4 # 线程数量
    #batch_size = [5, 10, 15, 20]  # 一次训练所选取的样本数
    #lr = [0.002, 0.004, 0.006, 0.008, 0.01]  # 学习率
    #nepoch = [40, 60, 80, 100]  # 训练的次数
    batch_size = 8
    lr = 0.004
    nepoch = 40
    k_fold = 5
    train_dataset, val_dataset = get_kfold(dataset_dir, k_fold, 1)
    train(dataset=train_dataset, model_dir=model_dir, workers=workers, batch_size=batch_size, lr=lr, nepoch=nepoch)
