"""
@author : Muhamad Irvan Dimetrio
NIM     : 18360018
Teknik Informatika
Institut Sains dan Teknologi Nasional
"""
from sklearn import datasets
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# Mengubah menjadi dataframe
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['species'])

# Menghapus spasi dari kolom
iris.columns = iris.columns.str.replace(' ', '') # Menghilangkan spasi pada header kolom
iris.head() # Update nama header kolom yang baru tanpa spasi

X = iris.iloc[:,:3] # variabel independe, dimana data mulai dari kolom 0 sampai 3
y = iris.species # variabel dependen , dimana adalah label kolom spesies
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

K = range(1, 10)
KM = [KMeans(n_clusters=k).fit(X) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D, axis=1) for D in D_k]
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d)/X.shape[0] for d in dist]

#Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X)**2)/X.shape[0]
bss = tss-wcss
varExplained = bss/tss*100

kIdx = 10-1
# Plot #
kIdx = 2

# Elbow curve
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(K, avgWithinSS, 'b*-')
plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

plt.subplot(1, 2, 2)
plt.plot(K, varExplained, 'b*-')
plt.plot(K[kIdx], varExplained[kIdx], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')

plt.tight_layout()
plt.show()