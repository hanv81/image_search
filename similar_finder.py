from torch.utils.data import DataLoader
from torchvision import transforms
from img2vec_pytorch import Img2Vec
from dataset import ImageDataset
from imutils import paths
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import faiss
import os
import shutil

class SimilarController:
    def __init__(self, image_path, batch_size):
        self.image_path = image_path
        self.i2v = Img2Vec()
        self.dataloader = DataLoader(ImageDataset(image_path), batch_size=32)

    def __load_image_vectors(self):
        vectors = pickle.loads(open("vectors.pickle", "rb").read())
        return vectors

    def __create_image_vectors(self):
        print('Creating image vectors')
        to_pil = transforms.ToPILImage()
        vectors = None
        for images in tqdm(self.dataloader, desc='Processing batch'):
            images = [to_pil(image) for image in images]
            if vectors is None:
                vectors = self.i2v.get_vec(images)
            else:
                vectors = np.concatenate((vectors, self.i2v.get_vec(images)), axis = 0)

        f = open("vectors.pickle", "wb")
        f.write(pickle.dumps(vectors))
        f.close()
        return vectors

    def __create_faiss_index(self):
        if os.path.exists('vectors.pickle'):
            vectors = self.__load_image_vectors()
        else:
            vectors = self.__create_image_vectors()

        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def find_similar(self, query_path, k = 4):
        self.__create_faiss_index()

        img_paths = list(paths.list_images(self.image_path))
        query_img_paths = list(paths.list_images(query_path))
        images = [Image.open(path) for path in query_img_paths]
        vectors = self.i2v.get_vec(images)
        faiss.normalize_L2(vectors)
        D, I = self.index.search(vectors, k)

        data = {'query_image': [], 'similar_image': [], 'similarity': []}
        for i in tqdm(range(len(D)), desc='Similar search result'):
            query_path = query_img_paths[i]
            query_filename = os.path.basename(query_path)
            if not os.path.exists(query_path + '_'):
                os.mkdir(query_path + '_')
            # print(f'Similar image of {query_path}')
            for j in range(k):
                # print(f'\t {img_paths[I[i][j]]} {D[i][j]}')
                shutil.copy(img_paths[I[i][j]], query_path + '_')
                if query_filename in data['query_image']:
                    data['query_image'].append('')
                else:
                    data['query_image'].append(query_filename)
                data['similar_image'].append(os.path.basename(img_paths[I[i][j]]))
                data['similarity'].append(D[i][j])

        df = pd.DataFrame(data)
        df.to_csv('similar.csv')

class SimilarFinder:
    def __init__(self, image_path, batch_size):
        self.controller = SimilarController(image_path, batch_size)

    def find_similar(self, query_path):
        self.controller.find_similar(query_path)