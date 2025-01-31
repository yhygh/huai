# Example with K-Means
from sklearn.cluster import KMeans
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)   # y: NumPy array, sr: sampling rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    return np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1)
    ])


data = []
files = []

audio_folder = './audio/m4a'

for file_name in os.listdir(audio_folder):
    file_path = os.path.join(audio_folder, file_name)
    features = extract_features(file_path)
    data.append(features)
    files.append(file_name)
    print(f'filename={file_name}')

data = np.array(data)

# Perform clustering
n_clusters = 3  # Choose a reasonable number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data)


# Example: Mapping files to clusters
file_cluster_mapping = {file: cluster for file, cluster in zip(files, clusters)}

print(f'\n----------------------------------')
# Print files grouped by cluster
for cluster_id in set(clusters):
    print(f"\nCluster {cluster_id}:")
    cluster_files = [file for file, cluster in file_cluster_mapping.items() if cluster == cluster_id]
    print(cluster_files)

labels = kmeans.labels_
print(f'labels = {labels}')
# labels[0] = 'wang'
# labels[1] = 'wang'
# labels[2] = 'zhao'
# labels[3] = 'zhao'
# print(f'labels = {labels}')

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# Visualize clusters
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    cluster_points = reduced_data[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
plt.title('Audio Clusters')
plt.legend()
plt.show()

# train a classifier for your genres

X_train, X_test, y_train, y_test = train_test_split(data, clusters, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Test the model
accuracy = clf.score(X_test, y_test)
print(f"Classifier Accuracy: {accuracy * 100:.2f}%")
# predict


new_file = './predict/wang_test.m4a'
new_features = extract_features(new_file).reshape(1, -1)
predicted_cluster = clf.predict(new_features)

print(f"Predicted Genre Cluster: {predicted_cluster[0]}")