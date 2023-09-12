import cv2
import numpy as np

# 读取彩色图像
img = cv2.imread('D:\\project\\deep_learning_recovery\\deep-phase-imaging\\0_train-cnn\\input\\OIP.jpg')

# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 将灰度图像reshape为(1, H, W)
gray = np.expand_dims(gray, axis=0)

# 保存为npy文件
np.save('D:\\project\\deep_learning_recovery\\deep-phase-imaging\\0_train-cnn\\input\\images.npy', gray)

# 对灰度图像进行傅里叶变换
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)

# 提取傅里叶变换结果的相位信息
magnitude_spectrum = 20 * np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)

# 将相位图像中的弧度转换为角度
phase_degrees = np.degrees(phase_spectrum)

# 显示相位角度图像
cv2.imshow('Phase Angle Image', phase_degrees)
cv2.waitKey(0)
cv2.destroyAllWindows()