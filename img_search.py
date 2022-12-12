from concurrent.futures import ThreadPoolExecutor
from imutils import paths
import numpy as np
import cv2
import time
import vptree
import pickle

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

def batch_hashing(hashes, file_paths):
    for path in file_paths:
        image = cv2.imread(path)
        h = dhash(image)
        h = convert_hash(h)
        p = hashes.get(h, [])
        p.append(path)
        hashes[h] = p

def create_hash():
    print("Hashing images...")
    t = time.time()
    batch_size = 100
    hashes = {}
    imagePaths = list(paths.list_images("images"))
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

def search_near_duplicate(hashes, tree):
    for h in hashes.keys():
        results = tree.get_all_in_range(h, 4)
        if len(results) > 1:
            print(f'{hashes[h]} have near-duplicate:')
            for d, hh in results:
                if d > 0:
                    print(f'\t{hashes[hh]} {d}')

if __name__ == "__main__":
    # hashes = create_hash()
    # tree = build_vptree(hashes)

    hashes, tree = load_hashes_and_vptree()
    imagePaths = list(paths.list_images("full"))
    search_near_duplicate(hashes, tree)