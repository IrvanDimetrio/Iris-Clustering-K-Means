"""
@author : Muhamad Irvan Dimetrio
NIM     : 18360018
Teknik Informatika
Institut Sains dan Teknologi Nasional
"""
from sklearn import datasets, metrics
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

# Mengubah menjadi dataframe
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['species'])

print("Data awal iris : ")
print(iris) # Cetak header kolom sebelum dirubah spasi ( name kolom masih pake spasi)

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
print("Label hasil clustering K-Means adalah : ")
print(model.labels_) # Cetak label jasil pembuatan oleh K-Means

# Menambahkan data ke dataframe yang ada untuk kolom pred_species hasil clustering K-Means
iris['pred_species'] = np.choose(model.labels_, [1, 0 , 2]).astype(np.int64)

# Mencatak akurasi prediksi clustering dan laporan klasifikasi
print("Akurasi : ",metrics.accuracy_score(iris.species, iris.pred_species))
print("Laporan Klasifikasi : ", metrics.classification_report(iris.species, iris.pred_species))

print("Cetak data iris dengan full rows : ")
pd.set_option('display.max_rows', None) # Menghilangkan default cetak max rows dari pandas
print(iris) # Mencetak semua baris data , bandingkan kolom spesies (aktual) dengan pred_species (hasil cluster)

print("Cetak data iris dengan sampel acak sebesar 0,3 : ")
print(iris.sample(frac=0.3))