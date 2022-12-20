from torch.utils.data import DataLoader
from torchvision import transforms
from img2vec_pytorch import Img2Vec
from dataset import ImageDataset
from duplicate_finder import DuplicateFinder
from imutils import paths
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import faiss
import os
import shutil
import argparse
import pandas as pd

def load_image_vectors():
    vectors = pickle.loads(open("vectors.pickle", "rb").read())
    return vectors

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
    parser.add_argument('--image_path', type = str, default = 'images')
    parser.add_argument('--query_path', type = str, default = 'query')
    parser.add_argument('--hash_size', type = int, default = 8)
    parser.add_argument('--batch_size', type = int, default = 32)
    args = parser.parse_known_args()[0]
    return args

def main():
    args = parse_args()
    if args.mode == 'duplicate':
        finder = DuplicateFinder(args.hash_size)
        finder.find_duplicate(args.image_path, args.batch_size)
    else:
        if os.path.exists('vectors.pickle'):
            vectors = load_image_vectors()
        else:
            vectors = create_image_vectors(args.image_path)

        index = create_faiss_index(vectors)
        search_similar(index, args.image_path, args.query_path)

if __name__ == "__main__":
    main()