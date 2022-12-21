import os
import imagehash
import pickle
import shutil
import time
import pandas as pd
from imutils import paths
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import deque
from vptree import VPTree
from PIL import Image

class HashLib:
    def __init__(self, hash_size):
        self.hash_size = hash_size
    
    def get_hash(self, path):
        image = Image.open(path)
        return imagehash.dhash(image, self.hash_size)

class HashController:
    def __init__(self, hasher):
        self.hasher = hasher
    
    def __load_hashes_and_vptree(self):
        self.tree = pickle.loads(open("vptree.pickle", "rb").read())
        self.hashes = pickle.loads(open("hashes.pickle", "rb").read())

    def __batch_hashing(self, file_paths, fr, to):
        to = min(len(file_paths), to)
        for i in range(fr, to):
            h = self.hasher.get_hash(file_paths[i])
            p = self.hashes.get(h, [])
            p.append(i)
            self.hashes[h] = p

    def __create_hashes(self, path, batch_size):
        # t = time.time()
        self.hashes = {}
        image_paths = list(paths.list_images(path))
        with ThreadPoolExecutor() as executor:
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Hashing images..."):
                executor.submit(self.__batch_hashing, image_paths, i, i + batch_size)
        
        # t = int(time.time()-t)
        # print(t)
        f = open("hashes.pickle", "wb")
        f.write(pickle.dumps(self.hashes))
        f.close()

    def _hamming(self, a, b):
        return a-b

    def __build_vptree(self):
        print("Building VP-Tree...")
        points = list(self.hashes.keys())
        self.tree = VPTree(points, self._hamming)

        f = open("vptree.pickle", "wb")
        f.write(pickle.dumps(self.tree))
        f.close()

    def _search_duplicate(self, path, batch_size):
        if os.path.exists('vptree.pickle'):
            self.__load_hashes_and_vptree()
        else:
            self.__create_hashes(path, batch_size)
            self.__build_vptree()

        if not os.path.exists('duplicate'):
            os.mkdir('duplicate')

        result = {}
        for h in tqdm(self.hashes.keys(), desc='Searching duplicate ...'):
            neighbors = self.tree.get_all_in_range(h, 4)
            if len(neighbors) > 1:
                i = self.hashes[h][0]
                result[i] = self.hashes[h][1:]

                for _, hh in neighbors:
                    for j in self.hashes[hh]:
                        if j not in self.hashes[h]:
                            result[i].append(j)

        return result

    def _aggregate_result(self, input):
        result = []
        visited = set()
        for key in tqdm(input, desc='Aggregating result ...'):
            if key in visited:
                continue

            queue = deque([key])
            reachable = set()

            while queue:
                curr_key = queue.popleft()
                reachable.add(curr_key)
                for value in input[curr_key]:
                    if value not in visited:
                        queue.append(value)
                visited.add(curr_key)

            result.append(reachable)

        return result

    def _export_excel(self, result, path):
        image_paths = list(paths.list_images(path))
        data = {'image': [], 'duplicate_image': []}
        for s in result:
            s = list(s)
            data['image'].append(os.path.basename(image_paths[s[0]]))
            for i in range(1, len(s)):
                if i > 1:
                    data['image'].append('')
                data['duplicate_image'].append(os.path.basename(image_paths[s[i]]))

        df = pd.DataFrame(data)
        df.to_csv('duplicate.csv')

class DuplicateFinder:
    def __init__(self, hash_size):
        hasher = HashLib(hash_size)
        self.hash_controller = HashController(hasher)

    def find_duplicate(self, path, batch_size):
        result = self.hash_controller._search_duplicate(path, batch_size)
        result = self.hash_controller._aggregate_result(result)
        self.hash_controller._export_excel(result, path)