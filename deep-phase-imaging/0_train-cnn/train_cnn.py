from get_kfold_data import get_kfold
import torch
import torch.utils.data as data
import subfunctions as sfns
import numpy as np
from trainer import train
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras import backend as K


# probe has already given
PROBE_PATH = 'D:\\project\\deep_learning_recovery\\deep-phase-imaging\\0_train-cnn\\input\\probe.npy'
# replace with your image dataset: (B, H, W)
IMAGES_PATH = 'D:\\project\\deep_learning_recovery\\deep-phase-imaging\\0_train-cnn\\input\\images.npy'
WEIGHTS_PATH = './0_train-cnn/output/weights'
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
DR_RATE = 0.2
model_dir = './model/'  # 网络参数保存位置
workers = 4  # 线程数量
batch_size = 8
k_fold = 5


def gen_masks(probe):
  p = np.abs(np.conj(probe)*probe)
  mask_small = p > np.max(p)*0.16
  mask_zoom = mask_small[12:20, 12:20]
  mask = sfns.imresize_big(mask_zoom, 4)
  return mask_small, mask_zoom, mask


def calc_sim_data(probe, imgs, mask, thickness):
  sim_x = np.zeros((imgs.shape[0], probe.shape[0], probe.shape[1]))
  sim_y = np.zeros((imgs.shape[0], mask.shape[0], mask.shape[1]))
  thickness_img = np.zeros((imgs.shape[0], probe.shape[0], probe.shape[1]))
  for i in range(0, imgs.shape[0]):
    thickness_i = thickness[i]
    obj = imgs[i, :, :]
    dp = np.exp(1.j*obj)*probe
    dp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(dp)))
    dp = np.abs(np.conj(dp)*dp)
    dp = np.random.poisson(dp)
    dp = np.sqrt(dp)
    # 扩展厚度图像到与强度图像相同的大小
    thickness_img[i, :, :] = np.ones_like(obj) * thickness_i
    # 将厚度图像和强度图像拼接在一起作为模型的输入
    sim_x[i, :, :] = dp
    obj = obj*mask
    obj = obj[12:20, 12:20]
    obj = sfns.imresize_big(obj, 4)
    sim_y[i, :, :] = obj
  return sim_x, sim_y, thickness_img


def l1_in_mask(y_true, y_pred):
  mask_tensor = torch.tensor(mask).cuda()
  error = torch.multiply(y_true[:, 0, :, :] - y_pred[:, 0, :, :], mask_tensor)
  error = torch.abs(error)
  return torch.sum(error) / torch.sum(mask_tensor)


def main():
  #--------------------------------------------------
  print("Loading probe and stock images")
  # todo: probe and imgs should be [B, num_slice, H, W]
  probe = np.load(PROBE_PATH)
  imgs = np.load(IMAGES_PATH)

  #--------------------------------------------------
  print("Generating masks")
  global mask
  mask_small, mask_zoom, mask = gen_masks(probe)
  np.save('D:\\project\\deep_learning_recovery\\deep-phase-imaging\\0_train-cnn\\output\\mask.npy', mask)
  mask = 1.*mask

  #--------------------------------------------------
  print("Generating simulated diffraction patterns")
  # todo: change into mutilayer slice
  thickness = np.random.rand(imgs.shape[0])
  sim_x, sim_y, thickness_img = calc_sim_data(probe, imgs, mask_small, thickness)
  sim_x = torch.tensor(sim_x).unsqueeze(dim=1)  # 衍射强度图
  sim_y = torch.tensor(sim_y).unsqueeze(dim=1)  # obj相位图作为ground truth
  thickness_img = torch.tensor(thickness_img).unsqueeze(dim=1)
  input_sim_x = np.concatenate([thickness_img, sim_x], axis=1)

  # 先转换成torch能识别的Dataset
  input_sim_x = torch.tensor(input_sim_x)
  # 将输入数据转换为 cuda 单精度浮点数张量
  input_sim_x = input_sim_x.to(torch.float32)
  sim_y = sim_y.to(torch.float32)
  dataset = data.TensorDataset(input_sim_x, sim_y)

  print("Training neural network")

  Lossfuc = l1_in_mask  # loss计算方法，自定义
  train_dataset, val_dataset = get_kfold(dataset, k_fold, 1)
  train(dataset=train_dataset, model_dir=model_dir, workers=workers, batch_size=batch_size, lr=LEARNING_RATE,
        nepoch=NUM_EPOCHS, dr_rate=DR_RATE, Lossfuc=Lossfuc)

  # '''tensorflow model'''
  # print("Training neural network")
  # model = sfns.create_model(DR_RATE)
  # opt = optimizers.SGD(learning_rate=LEARNING_RATE)
  # model.compile(optimizer=opt, loss=l1_in_mask)
  # model.summary()
  # checkpoints = callbacks.ModelCheckpoint('%s/{epoch:03d}.hdf5' %WEIGHTS_PATH,
  #   save_weights_only=False, verbose=1, save_freq="epoch")
  # history = model.fit(input_sim_x, sim_y, shuffle=True, batch_size=16, verbose=1,
  #   epochs=NUM_EPOCHS, validation_split=0.05, callbacks=[checkpoints])


if __name__ == "__main__":
  main()

