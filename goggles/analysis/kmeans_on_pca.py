def train_kmeans_on_pca(train_data, train_labels, test_data, test_labels, n_components=2):
    from sklearn.decomposition import KernelPCA
    from sklearn.cluster import KMeans

    pca = KernelPCA(n_components=n_components).fit(train_data)
    
    transformed_train_data = pca.transform(train_data)
    transformed_test_data = pca.transform(test_data)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(transformed_train_data)
    
    kmeans_pred = kmeans.predict(transformed_test_data)    
    kmeans_pred[kmeans_pred == 0] = -1
    
    acc = accuracy_score(test_labels, kmeans_pred)
    
    return pca, kmeans, max(acc, 1-acc)

def visualization_kmeans_on_pca(pca, kmeans, train_data, test_data, train_labels, test_labels):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    transformed_test_data = pca.transform(test_data)
    
    kmeans_pred = kmeans.predict(transformed_test_data)    
    kmeans_pred[kmeans_pred == 0] = -1
    
    
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue']
    lw = 2
    test_labels = np.array(test_labels)
    
    for color, i, target_name in zip(colors, [1, -1], ['species_1', 'species_2']):
        plt.scatter(transformed_test_data[test_labels == i, 0], transformed_test_data[test_labels == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')
    
    

    plt.subplot(1, 2, 2)
    plt.scatter(transformed_test_data[:, 0], transformed_test_data[:, 1], c=kmeans_pred)
    plt.title("KMeans Clusters")
    
    plt.show()
