
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
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

def get_eigen_topics_ngrams(docs,  ngram_range=(2,2), v_1=0, v_2=10):
    '''
    docs: list of strings
    '''
    vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
    print(vectorizer)
    vectors = vectorizer.fit_transform(docs).todense()
    U, s, VT = linalg.svd(vectors, full_matrices=False)
    vocab = np.array(vectorizer.get_feature_names())
    eigen_topics = show_topics(VT[v_1:v_2], vocab=vocab)

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


def evidence_lowerbound_svd(X, k):
    '''

    X: np.ndarray
    k: int
    '''
    U, s, VT = linalg.svd(X, full_matrices=False)
    return np.sum(s[:k]**2) / X.shape[0]

def get_best_k(docs, r_1=2, r_2=10, model_name = 'paraphrase-MiniLM-L6-v2', device='cpu'):
    '''
    search with elbo method
    docs: list of strings
    '''
    model = SentenceTransformer(model_name, device= device)
    embeddings = model.encode(docs)
    elbos = []
    for k in range(r_1, r_2):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        elbos.append(elbo(kmeans, embeddings, k))
    return np.argmax(elbos) + 2

def get_best_k_svd(docs, r_1=2, r_2=10):
    '''
    search with svd method
    docs: list of strings
    '''
    vectorizer = CountVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(docs).todense()
    svds = []
    for k in range(r_1, r_2):
        svds.append(evidence_lowerbound_svd(vectors, k))
    return np.argmax(svds) + 2