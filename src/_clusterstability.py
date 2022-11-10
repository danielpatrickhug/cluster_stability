from io import BytesIO  
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm, svd
from PIL import Image
import requests as r
from sklearn.cluster import KMeans


def query_lexica(query, num_images=5):
    '''
    query: string
    num_images: int
    '''
    resp = r.get(f"https://lexica.art/api/v1/search?q={query}", timeout=5)
    images = []
    for img in resp['images'][:num_images]:
        url = img['src']
        response = r.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        images.append(img)
    return images


def image_grid(imgs, rows, cols):
    '''
    imgs: list of PIL images
    rows: int
    cols: int
    '''
    width,height = imgs[0].size
    grid = Image.new('RGB', size=(cols*width, rows*height))
    for i, img in enumerate(imgs): 
        grid.paste(img, box=(i%cols*width, i//cols*height))
    return grid

def cosine_similarity(emb_a:np.ndarray, emb_b:np.ndarray) -> float:
    '''
    a: np.ndarray
    b: np.ndarray
    '''
    return np.dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b))


def show_topics(a, vocab, num_top_words=8):
    """
    a: np.ndarray
    vocab: np.ndarray
    num_top_words: int
    """
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]

def get_eigen_topics(docs):
    '''
    model: SentenceTransformer
    docs: list of strings
    '''
    vectorizer = CountVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(docs).todense()
    _, _, VT = svd(vectors, full_matrices=False)
    vocab = np.array(vectorizer.get_feature_names())
    eigen_topics = show_topics(VT[:10], vocab)
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