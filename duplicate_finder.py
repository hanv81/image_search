import os
import imagehash
import pickle
import shutil
import time
from imutils import paths
from concurrent.futures import ThreadPoolExecutor
from vptree import VPTree
from PIL import Image

class HashLib:
    def __init__(self, hash_size):
        self.hash_size = hash_size
    
    def get_hash(self, path):
        image = Image.open(path)
        return imagehash.dhash(image, self.hash_size)

class HashController:
    def __init__(self, hasher) -> None:
        self.hasher = hasher
    
    def __load_hashes_and_vptree(self):
        self.tree = pickle.loads(open("vptree.pickle", "rb").read())
        self.hashes = pickle.loads(open("hashes.pickle", "rb").read())

    def __batch_hashing(self, file_paths):
        for path in file_paths:
            h = self.hasher.get_hash(path)
            p = self.hashes.get(h, [])
            p.append(path)
            self.hashes[h] = p

    def __create_hashes(self, path, batch_size):
        print("Hashing images...")
        t = time.time()
        self.hashes = {}
        image_paths = list(paths.list_images(path))
        with ThreadPoolExecutor() as executor:
            for i in range(0, len(image_paths), batch_size):
                executor.submit(self.__batch_hashing, image_paths[i : i + batch_size])
        
        t = int(time.time()-t)
        print(t)
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

    def search_duplicate(self, path, batch_size):
        if os.path.exists('vptree.pickle'):
            self.__load_hashes_and_vptree()
        else:
            self.__create_hashes(path, batch_size)
            self.__build_vptree()

        if not os.path.exists('duplicate'):
            os.mkdir('duplicate')
        for h in self.hashes.keys():
            results = self.tree.get_all_in_range(h, 4)
            if len(results) > 1:
                duplicate_path = os.path.join('duplicate', os.path.basename(self.hashes[h][0]))
                if not os.path.exists(duplicate_path):
                    os.mkdir(duplicate_path)
                shutil.copy(self.hashes[h][0], duplicate_path)
                print(f'{self.hashes[h]} have duplicate:')
                for d, hh in results:
                    if d > 0:
                        print(f'\t{self.hashes[hh]} {d}')
                        shutil.copy(self.hashes[hh][0], duplicate_path)

class DuplicateFinder:
    def __init__(self, hash_size):
        hasher = HashLib(hash_size)
        self.hash_controller = HashController(hasher)

    def find_duplicate(self, path, hash_size):
        self.hash_controller.search_duplicate(path, hash_size)

    def _aggregate_result(self):
        pass

    def _export_excel(self):
        pass