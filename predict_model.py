#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# В случае, если ошибка с тензором
# !pip install tensorly
# !pip install opencv-python
# !pip install --force-reinstall -v "tensorly==0.7.0"
# !pip install --force-reinstall -v "numpy==1.23.5"


# In[ ]:


import logging
import os
import warnings
from glob import glob
from random import shuffle

import cv2
import numpy as np
import pandas as pd
import scipy as sp
import scipy.misc
import tensorly as tl
from tensorly.decomposition import tucker, partial_tucker
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.linear_model import orthogonal_mp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from typing import Type

import pickle

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)


# In[ ]:


class Pursuit:
    """
    Algorithms that inherit from this class are methods to solve problems of the like
    \min_A \| DA - Y \|_2 s.t. \|A\|_0 <= t.
    Here, D is a given dictionary of size (n x K)
    Y is a given matrix of size (n x N), where N is the number of samples
    The Pursuit will return a matrix A of size (K x N).
    """

    def __init__(self, dictionary, max_iter=False, tol=None, sparsity=None):
        self.D = np.array(dictionary.matrix)
        self.max_iter = max_iter
        self.tol = tol
        self.sparsity = sparsity
        if (self.tol is None and self.sparsity is None) or (self.tol is not None and self.sparsity is not None):
            raise ValueError("blub")
        self.data = None
        self.alphas = []

    def fit(self, Y):
        return [], self.alphas
    

class OrthogonalMatchingPursuit(Pursuit):
    """
    Wrapper for orthogonal_mp from scikit-learn
    """

    def fit(self, Y):
        return orthogonal_mp(self.D, Y, n_nonzero_coefs=self.sparsity,
                             tol=self.tol, precompute=True)


# In[ ]:


class Dictionary:
    """
    The Dictionary class is more or less a wrapper around the numpy array class. It holds a numpy ndarray in
    the attribute `matrix` and adds some useful functions for it. The dictionary elements can be accessed
    either by D.matrix[i,j] or directly through D[i,j].
    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.shape = matrix.shape

    def __getitem__(self, item):
        return self.matrix[item]

    def is_unitary(self):
        """
        Checks whether the dictionary is unitary.
        Returns:
            True, if the dicitonary is unitary.
        """
        n, K = self.shape
        if n == K:
            return np.allclose(np.dot(self.matrix.T, self.matrix), np.eye(n))
        else:
            return False

    def is_normalized(self):
        """
        Checks wheter the dictionary is l2-normalized.
        Returns:
            True, if dictionary is l2-normalized.
        """
        n, K = self.shape
        return np.allclose([np.linalg.norm(self.matrix[:, i]) for i in range(K)], np.ones(K))


    def mutual_coherence(self):
        """
        Computes the dictionary's mutual coherence.
        Returns:
            Mutual coherence
        """
        return np.max(self._mutual_coherence(self.matrix))

    @staticmethod
    def _mutual_coherence(D):
        n, K = D.shape
        mu = [np.abs(np.dot(D[:, i].T, D[:, j]) /
                     (np.linalg.norm(D[:, i]) * np.linalg.norm(D[:, j])))
              for i in range(K) for j in range(K) if j != i]
        return mu

    def to_img(self):
        """
        Transforms the dictionary columns into patches and orders them for plotting purposes.
        Returns:
            Reordered dictionary matrix
        """
        # dictionary dimensions
        D = self.matrix
        n, K = D.shape
        M = self.matrix
        # stretch atoms
        for k in range(K):
            M[:, k] = M[:, k] - (M[:, k].min())
            if M[:, k].max():
                M[:, k] = M[:, k] / D[:, k].max()

        # patch size
        n_r = int(np.sqrt(n))

        # patches per row / column
        K_r = int(np.sqrt(K))

        # we need n_r*K_r+K_r+1 pixels in each direction
        dim = n_r * K_r + K_r + 1
        V = np.ones((dim, dim)) * np.min(D)

        # compute the patches
        patches = [np.reshape(D[:, i], (n_r, n_r)) for i in range(K)]

        # place patches
        for i in range(K_r):
            for j in range(K_r):
                V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
                    i * K_r + j]
        return V


# In[ ]:


class KSVD:
    """
    Implements the original K-SVD Algorithm as described in [1].
    [1] Aharon, M., Elad, M. and Bruckstein, A., 2006. K-SVD: An algorithm for designing overcomplete dictionaries for
        sparse representation. IEEE Transactions on signal processing, 54(11), p.4311.
    Args:
        dictionary: Initial dictionary of type sparselandtools.dictionaries.Dictionary
        pursuit: Pursuit method to be used (any method from sparselandtools.pursuits)
        sparsity: Target sparsity
        noise_gain: Target noise_gain. If set, this will override the target sparsity
        sigma: Signal or image noise standard deviation.
    """

    def __init__(self, dictionary: Dictionary, pursuit: Type[Pursuit], sparsity: int, noise_gain=None, sigma=None):
        self.dictionary = Dictionary(dictionary.matrix)
        self.alphas = None
        self.pursuit = pursuit
        self.sparsity = sparsity
        self.noise_gain = noise_gain
        self.sigma = sigma
        self.original_image = None
        self.sparsity_values = []
        self.mses = []
        self.ssims = []
        self.psnrs = []
        self.iter = None

    def sparse_coding(self, Y: np.ndarray):
        logging.info("Entering sparse coding stage...")
        if self.noise_gain and self.sigma:
            p = self.pursuit(self.dictionary, tol=(self.noise_gain * self.sigma))
        else:
            p = self.pursuit(self.dictionary, sparsity=self.sparsity)
        self.alphas = p.fit(Y)
        logging.info("Sparse coding stage ended.")

    def dictionary_update(self, Y: np.ndarray):
        # iterate rows
        D = self.dictionary.matrix
        n, K = D.shape
        R = Y - D.dot(self.alphas)
        for k in range(K):
            logging.info("Updating column %s" % k)
            wk = np.nonzero(self.alphas[k, :])[0]
            if len(wk) == 0:
                continue
            Ri = R[:,wk] + D[:,k,None].dot(self.alphas[None,k,wk])
            U, s, Vh = np.linalg.svd(Ri)
            D[:, k] = U[:, 0]
            self.alphas[k, wk] = s[0] * Vh[0, :]
            R[:, wk] = Ri - D[:,k,None].dot(self.alphas[None,k,wk])
        self.dictionary = Dictionary(D)

    def fit(self, Y: np.ndarray, iter: int):
        for i in range(iter):
            logging.info("Start iteration %s" % (i + 1))
            self.sparse_coding(Y)
            self.dictionary_update(Y)
        return self.dictionary, self.alphas


# In[ ]:


def preprocess(image):
    core, factors = tucker(tl.tensor(image), rank=image.shape)
    return factors[1]


def get_matrix(y, image_path, patch_size):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    component = preprocess(image)
    
    # Извлеките все патчи изображения
    patches = extract_patches_2d(component, (patch_size, patch_size))
    
    # Получите количество патчей
    num_patches = patches.shape[0]
    
    # Рассчитайте стандартное отклонение (в качестве меры вариации) для каждого патча
    std_devs = np.std(patches, axis=(1, 2))
    
    # Найдите индексы 5 патчей с наибольшей вариацией
    top_indices = np.argsort(std_devs)[-5:]
    
    # Выберите 5 случайных патчей, которые не пересекаются с патчами с наибольшей вариацией
    random_indices = []
    while len(random_indices) < 5:
        index = random.randint(0, num_patches - 1)
        if index not in top_indices:
            random_indices.append(index)
    
    # Получите случайные и патчи с наибольшей вариацией
    random_patches = patches[random_indices]
    top_patches = patches[top_indices]
    
    # Объедините случайные и патчи с наибольшей вариацией
    selected_patches = np.concatenate((random_patches, top_patches), axis=0)
    
    # Преобразуйте патчи в нужный формат данных
    data = selected_patches.reshape(selected_patches.shape[0], -1)
    
    # Обновите y с добавлением выбранных патчей
    y = np.vstack([y, data])
    
    return y


# In[ ]:


# Задать необходимые переменные 

# Директория с изображениями
image_directory = 'images/'
# Путь ко всем изображениям
image_paths = glob(os.path.join(image_directory, '*.png'))

# Если необходимо, то перемешать
# shuffle(image_paths)

# Размер патча
patch_size = 10
# Количество итераций обучения
iterations = 1
# Файл с временем цветения и изображением
response_data = 'response.csv'
# Название файла словаря
dict_name = 'dictionary.npy'
# Названия файла с фичами
features_name = 'dataset.csv'


# In[ ]:


# Слияние изображений
y = np.zeros(patch_size * patch_size)
for i, image_path in enumerate(image_paths):
    print('%d/%d %s' % (i, len(image_paths), image_path))
    y = get_matrix(y, image_path, patch_size)

y = np.delete(y, 0, axis=0)


# In[ ]:


# Обучение словаря
u, s, v = np.linalg.svd(y.T)
initial_dictionary = Dictionary(u)
ksvd = KSVD(initial_dictionary, OrthogonalMatchingPursuit, patch_size * patch_size)
learn_dict, coeff = ksvd.fit(y.T, iterations)
v = learn_dict.matrix


# In[ ]:


# Отобразить полученный словарь
plt.imshow(learn_dict.to_img(), cmap='gray')


# In[ ]:


# Вычисление фич
def get_features(x):
    f_mic = []
    f_mac = []
    for i in range(x.shape[0]):
        values = x[i]
        values = np.abs(values[values!=0])
        sigma, _, mean = sp.stats.lognorm.fit(values, loc=0)
        f_mic.append(np.exp(mean + 0.5*sigma**2))
        f_mac.append(values.shape[0])
    return f_mic + f_mac


# In[ ]:


# Сохраняем словарь на диск
np.save(open(dict_name, 'wb'), v)
v = np.load(open(dict_name, 'rb'))


# In[ ]:


# Создание таблицы для записи фич
df = pd.read_csv(response_data)
images = df['image'].tolist()
labels = df['label'].tolist()
image_paths = [image_directory+image for image in images]
del df
columns = ['image']
for i in range(v.shape[1]):
    columns.append('f%d_mic'%(i+1))
for i in range(v.shape[1]):
    columns.append('f%d_mac'%(i+1))
columns.append('label')
data = {column: [] for column in columns}
df = pd.DataFrame(data)


# In[ ]:


# Извлечение фич
for i, image_path in enumerate(image_paths):
    y = np.zeros(patch_size * patch_size)
    y = get_matrix(y, image_path, patch_size)
    y = y.T
    X = orthogonal_mp(v, y, n_nonzero_coefs=patch_size*patch_size)
    features = get_features(X)
    data = [images[i]] + features + [labels[i]]
    data = {column: [datum] for column, datum in zip(columns, data)}
    tmp = pd.DataFrame(data)
    df = pd.concat([df, tmp], axis=0)
    print('%d/%d %s' % (i, len(image_paths), image_path))


# In[ ]:


# Сохранение таблицы с фичами
df.to_csv(features_name, index=False)


# In[ ]:


# Обучение модели
df = pd.read_csv(features_name)
columns = [column for column in df.columns if column not in ['image', 'label']]
x = df[columns].to_numpy()
y = df['label'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = make_pipeline(StandardScaler(), SVR(C=1000, epsilon=1, gamma=1))
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[ ]:


# Максимальная ошибка прогнозирования
max(abs(y_pred-y_test))


# In[ ]:


# Средняя ошибка прогнозирования
mean_absolute_error(y_test, y_pred)

