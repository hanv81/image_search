from torch.utils.data import Dataset
from torchvision.io import read_image
from imutils import paths

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(list(paths.list_images(self.img_dir)))

    def __getitem__(self, i):
        img_paths = list(paths.list_images(self.img_dir))
        image = read_image(img_paths[i])
        return image