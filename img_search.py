from torch.utils.data import DataLoader
from torchvision import transforms
from img2vec_pytorch import Img2Vec
from dataset import ImageDataset
from concurrent.futures import ThreadPoolExecutor
from imutils import paths
import numpy as np
import cv2
import time
import vptree
import pickle
import faiss
import os
import shutil
import argparse

def dhash(image, hashSize=8):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	h = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
	return h

def convert_hash(h):
	return int(np.array(h, dtype="float64"))

def hamming(a, b):
	return bin(int(a) ^ int(b)).count("1")

def hash_file(path):
    image = cv2.imread(path)
    h = dhash(image)
    return convert_hash(h)

def batch_hashing(hashes, file_paths):
    for path in file_paths:
        h = hash_file(path)
        p = hashes.get(h, [])
        p.append(path)
        hashes[h] = p

def create_hash(image_path):
    print("Hashing images...")
    t = time.time()
    batch_size = 100
    hashes = {}
    imagePaths = list(paths.list_images(image_path))
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(imagePaths), batch_size):
            executor.submit(batch_hashing, hashes, imagePaths[i : i + batch_size])

    t = int(time.time()-t)
    print(t)
    f = open("hashes.pickle", "wb")
    f.write(pickle.dumps(hashes))
    f.close()
    
    return hashes

def build_vptree(hashes):    
    print("Building VP-Tree...")
    points = list(hashes.keys())
    tree = vptree.VPTree(points, hamming)

    f = open("vptree.pickle", "wb")
    f.write(pickle.dumps(tree))
    f.close()
    return tree

def load_hashes_and_vptree():
    tree = pickle.loads(open("vptree.pickle", "rb").read())
    hashes = pickle.loads(open("hashes.pickle", "rb").read())
    return hashes, tree

def load_image_vectors():
    vectors = pickle.loads(open("vectors.pickle", "rb").read())
    return vectors

def search_near_duplicate(hashes, tree):
    for h in hashes.keys():
        results = tree.get_all_in_range(h, 4)
        if len(results) > 1:
            print(f'{hashes[h]} have near-duplicate:')
            for d, hh in results:
                if d > 0:
                    print(f'\t{hashes[hh]} {d}')

def create_image_vectors(image_path):
    i2v = Img2Vec()
    dataset = ImageDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=32)
    vectors = None
    img_paths = []
    for i, (images, list_paths) in enumerate(dataloader):
        print(f'Loading batch {i}')
        img_paths.extend(list_paths)
        images = [transforms.ToPILImage()(image) for image in images]
        if vectors is None:
            vectors = i2v.get_vec(images)
        else:
            vectors = np.concatenate((vectors, i2v.get_vec(images)), axis = 0)

    f = open("vectors.pickle", "wb")
    f.write(pickle.dumps(vectors))
    f.close()
    return vectors

def create_faiss_index(vectors):
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def search_similar(index, image_path, query_path):
    img_paths = list(paths.list_images(image_path))
    query_img_paths = list(paths.list_images(query_path))
    images = [transforms.ToPILImage()(cv2.imread(path)) for path in query_img_paths]
    vectors = Img2Vec().get_vec(images)
    faiss.normalize_L2(vectors)
    k = 4
    D, I = index.search(vectors, k)
    for i in range(len(D)):
        if not os.path.exists(query_img_paths[i]+'_'):
            os.mkdir(query_img_paths[i]+'_')
        print(f'Similar image of {query_img_paths[i]}')
        for j in range(k):
            print(f'\t {img_paths[I[i][j]]} {D[i][j]}')
            shutil.copy(img_paths[I[i][j]], query_img_paths[i]+'_')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'similar')
    parser.add_argument('--images', type = str, default = 'images')
    parser.add_argument('--query', type = str, default = 'query')
    args = parser.parse_known_args()[0]
    return args.mode, args.images, args.query

if __name__ == "__main__":
    mode, image_path, query_path = parse_args()
    if mode == 'duplicate':
        if os.path.exists('vptree.pickle'):
            hashes, tree = load_hashes_and_vptree()
        else:
            hashes = create_hash(image_path)
            tree = build_vptree(hashes)

        search_near_duplicate(hashes, tree)
    else:
        if os.path.exists('vectors.pickle'):
            vectors = load_image_vectors()
        else:
            vectors = create_image_vectors(image_path)

        index = create_faiss_index(vectors)
        search_similar(index, image_path, query_path)