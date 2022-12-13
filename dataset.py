from torch.utils.data import Dataset
from torchvision.io import read_image
from imutils import paths

class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.list_paths = list(paths.list_images(self.img_dir))

    def __len__(self):
        return len(self.list_paths)

    def __getitem__(self, i):
        path = self.list_paths[i]
        image = read_image(path)
        return image, path