from torch.utils.data import DataLoader
from torchvision import transforms
from img2vec_pytorch import Img2Vec
from dataset import ImageDataset
from concurrent.futures import ThreadPoolExecutor
from imutils import paths
from PIL import Image
from tqdm import tqdm
import imagehash
import numpy as np
import cv2
import time
import vptree
import pickle
import faiss
import os
import shutil
import argparse
import pandas as pd

def hamming(a, b):
	return a-b

def hash_file(path):
    image = Image.open(path)
    return imagehash.dhash(image, 8)

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
    image_paths = list(paths.list_images(image_path))
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(image_paths), batch_size):
            executor.submit(batch_hashing, hashes, image_paths[i : i + batch_size])

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

def search_duplicate(hashes, tree):
    if not os.path.exists('duplicate'):
        os.mkdir('duplicate')
    for h in hashes.keys():
        results = tree.get_all_in_range(h, 4)
        if len(results) > 1:
            duplicate_path = os.path.join('duplicate', os.path.basename(hashes[h][0]))
            if not os.path.exists(duplicate_path):
                os.mkdir(duplicate_path)
            shutil.copy(hashes[h][0], duplicate_path)
            print(f'{hashes[h]} have duplicate:')
            for d, hh in results:
                if d > 0:
                    print(f'\t{hashes[hh]} {d}')
                    shutil.copy(hashes[hh][0], duplicate_path)

def create_image_vectors(image_path):
    print('Creating image vectors')
    i2v = Img2Vec()
    to_pil = transforms.ToPILImage()
    vectors = None
    dataloader = DataLoader(ImageDataset(image_path), batch_size=32)
    for images in tqdm(dataloader, desc='Processing batch'):
        images = [to_pil(image) for image in images]
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
    images = [Image.open(path) for path in query_img_paths]
    vectors = Img2Vec().get_vec(images)
    faiss.normalize_L2(vectors)
    k = 4
    D, I = index.search(vectors, k)
    data = {'query_image': [], 'similar_image': [], 'similarity': []}
    for i in range(len(D)):
        query_path = query_img_paths[i]
        query_filename = os.path.basename(query_path)
        if not os.path.exists(query_path + '_'):
            os.mkdir(query_path + '_')
        print(f'Similar image of {query_path}')
        for j in range(k):
            if D[i][j] >= .8:
                continue
            print(f'\t {img_paths[I[i][j]]} {D[i][j]}')
            shutil.copy(img_paths[I[i][j]], query_path + '_')
            if query_filename in data['query_image']:
                data['query_image'].append('')
            else:
                data['query_image'].append(query_filename)
            data['similar_image'].append(os.path.basename(img_paths[I[i][j]]))
            data['similarity'].append(D[i][j])

    df = pd.DataFrame(data)
    df.to_csv('similar.csv')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'similar')
    parser.add_argument('--images', type = str, default = 'images')
    parser.add_argument('--query', type = str, default = 'query')
    args = parser.parse_known_args()[0]
    return args.mode, args.images, args.query

def main():
    mode, image_path, query_path = parse_args()
    if mode == 'duplicate':
        if os.path.exists('vptree.pickle'):
            hashes, tree = load_hashes_and_vptree()
        else:
            hashes = create_hash(image_path)
            tree = build_vptree(hashes)

        search_duplicate(hashes, tree)
    else:
        if os.path.exists('vectors.pickle'):
            vectors = load_image_vectors()
        else:
            vectors = create_image_vectors(image_path)

        index = create_faiss_index(vectors)
        search_similar(index, image_path, query_path)

if __name__ == "__main__":
    main()