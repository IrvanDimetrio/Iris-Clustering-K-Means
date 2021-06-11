"""
@author : Muhamad Irvan Dimetrio
NIM     : 18360018
Teknik Informatika
Institut Sains dan Teknologi Nasional
"""
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# K-means Cluster
model = KMeans(n_clusters=3, random_state=11) # Terapkan model ke algoritma K_Means
model.fit(X)

# Menambahkan data ke dataframe yang ada untuk kolom pred_species hasil clustering K-Means
iris['pred_species'] = np.choose(model.labels_, [1, 0 , 2]).astype(np.int64)

# Membuat ukuran gambar dengan sumbu 'ax1' berukuran 2x2
fig, ax1 = plt.subplots(2, 2, figsize=(
    22, 18), gridspec_kw={'hspace': 0.5, 'wspace': 0.2})

colorplot = dict({0.0: 'red', 0:'red', 1.0:'green', 2.0:'blue', 2:'blue'})

#Plot Sepal
sns.scatterplot(data = iris, x='sepallength(cm)', y='sepalwidth(cm)', hue='species',
                legend='full', ax=ax1[0, 0], palette=colorplot).set_title('Sepal (Actual)')
sns.scatterplot(data = iris, x='sepallength(cm)', y='sepalwidth(cm)', hue='pred_species',
                legend='full', ax=ax1[0, 1], palette=colorplot).set_title('Sepal (Predicted)')

#Plot Petal
sns.scatterplot(data = iris, x='petallength(cm)', y='petalwidth(cm)', hue='species',
                legend='full', ax=ax1[1, 0], palette=colorplot).set_title('Petal (Actual)')
sns.scatterplot(data = iris, x='petallength(cm)', y='petalwidth(cm)', hue='pred_species',
                legend='full', ax=ax1[1, 1], palette=colorplot).set_title('Petal (Predicted)')

# Show the figure of plot

plt.show()