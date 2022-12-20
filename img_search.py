from duplicate_finder import DuplicateFinder
from similar_finder import SimilarFinder
import argparse

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
        finder = SimilarFinder(args.image_path, args.batch_size)
        finder.find_similar(args.query_path)

if __name__ == "__main__":
    main()