
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
#from numpy.linalg import  svd
from scipy import linalg
from src._utils import show_topics, query_lexica, image_grid




def get_eigen_topics(docs):
    '''
    model: SentenceTransformer
    docs: list of strings
    '''
    vectorizer = CountVectorizer(stop_words='english')
    print(vectorizer)
    vectors = vectorizer.fit_transform(docs).todense()
    _, _, VT = linalg.svd(vectors, full_matrices=False)
    vocab = np.array(vectorizer.get_feature_names())
    eigen_topics = show_topics(VT[:10], vocab=vocab)
    return eigen_topics
    

def cluster_documents(docs, model_name = 'paraphrase-MiniLM-L6-v2', num_clusters=5, device='cpu'):
    '''
    docs: list of strings
    num_clusters: int
    '''
    model = SentenceTransformer(model_name, device= device)
    embeddings = model.encode(docs)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    clusters = [[] for _ in range(num_clusters)]
    for i, cluster in enumerate(kmeans.labels_):
        clusters[cluster].append(docs[i])
    return clusters

def get_cluster_images(docs, num_clusters=5, model_name = 'paraphrase-MiniLM-L6-v2', device='cpu'):
    '''
    docs: list of strings
    num_clusters: int
    '''
    clusters = cluster_documents(docs, num_clusters=num_clusters, model_name=model_name, device=device)
    topic_grids = {}
    for cluster in clusters:
        images = []
        topics = get_eigen_topics(cluster)
        for topic in topics:
            images.append(query_lexica(topic))
        topic_grids[tuple(topics)] = image_grid(images, 2, 4)
    return topic_grids

def elbo(model, X, k):
    '''
    model: sklearn.cluster.KMeans
    X: np.ndarray
    k: int
    '''
    return model.score(X) * X.shape[0] - k * model.inertia_

def get_best_k(docs, model_name = 'paraphrase-MiniLM-L6-v2', device='cpu'):
    '''
    docs: list of strings
    '''
    model = SentenceTransformer(model_name, device= device)
    embeddings = model.encode(docs)
    elbos = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        elbos.append(elbo(kmeans, embeddings, k))
    return np.argmax(elbos) + 2