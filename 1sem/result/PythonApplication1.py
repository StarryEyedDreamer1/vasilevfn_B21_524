import numpy as np
from PIL import Image as pim

def stretch_image(input_array, M):
    #���������� (������������) ����������� � M ���
    H, W = input_array.shape[:2]
    new_H = H * M
    new_W = W * M
    output_array = np.zeros((new_H, new_W, input_array.shape[2]), dtype=input_array.dtype)
    for y in range(new_H):
        for x in range(new_W):
            orig_y = y // M
            orig_x = x // M
            output_array[y, x] = input_array[orig_y, orig_x]
    return output_array

def compress_image(input_array, N):
    #������ (���������) ����������� � N ���
    H, W = input_array.shape[:2]
    new_H = H // N
    new_W = W // N
    output_array = np.zeros((new_H, new_W, input_array.shape[2]), dtype=input_array.dtype)
    for y in range(new_H):
        for x in range(new_W):
            orig_y = y * N
            orig_x = x * N
            output_array[y, x] = input_array[orig_y, orig_x]
    return output_array

def resample_image_two_pass(input_array, M, N):
    #����������������� ����������� � K=M/N ��� ���� ���������� � 
    #������������ ������ (� ��� �������)
    stretched_array = stretch_image(input_array, M)
    resampled_array = compress_image(stretched_array, N)
    return resampled_array

def resample_image_one_pass(input_array, K):
    #����������������� ����������� � K ��� �� ���� ������
    H, W = input_array.shape[:2]
    new_H = H * K
    new_H = round(new_H)
    new_W = W * K
    new_W = round(new_W)
    output_array = np.zeros((new_H, new_W, input_array.shape[2]), dtype=input_array.dtype)
    for y in range(new_H):
        for x in range(new_W):
            orig_y = int(y // K)
            orig_x = int(x // K)
            output_array[y, x] = input_array[orig_y, orig_x]
    return output_array

# �������� ��������� ����������� � �������������� ��� � ������ NumPy
input_image = pim.open('muar.png')
input_array = np.array(input_image)

# ��������� M � N ��� ������������� ����� M/N, ������������� ������ ��������� �����������
M, N = 10, 20 # 0.5

# ���������� ����������� � M ���
stretched_array = stretch_image(input_array, M)

# ������ ����������� � N ���
compressed_array = compress_image(input_array, N)

# ����������������� ����������� � K=M/N ��� ���� ���������� � ������������ ������ (� ��� �������)
resampled_array_two_pass = resample_image_two_pass(input_array, M, N)

# ����������������� ����������� � K ��� �� ���� ������
resampled_array_one_pass = resample_image_one_pass(input_array, M/N)

# �������������� �������� �������� ������� � �����������
stretched_image = pim.fromarray(stretched_array)
compressed_image = pim.fromarray(compressed_array)
resampled_image_two_pass = pim.fromarray(resampled_array_two_pass)
resampled_image_one_pass = pim.fromarray(resampled_array_one_pass)

# ���������� �������� �����������
stretched_image.save('stretched_image.png')
compressed_image.save('compressed_image.png')
resampled_image_two_pass.save('resampled_image_two_pass.png')
resampled_image_one_pass.save('resampled_image_one_pass.png')
