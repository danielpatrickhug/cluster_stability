from io import BytesIO  
import numpy as np 
from numpy.linalg import norm
from PIL import Image
import requests as r



def query_lexica(query, num_images=5):
    '''
    query: string
    num_images: int
    '''
    resp = r.get(f"https://lexica.art/api/v1/search?q={query}", timeout=5)
    images = []
    for img in resp['images']:
        if len(images) == num_images:
            break
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

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    #Spherically interpolate between two embedding vectors

    if not isinstance(v0, np.ndarray):
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1

    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    return v2