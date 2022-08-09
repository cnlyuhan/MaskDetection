from re import I
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from utils.utils import collate_fn, generate_target
from torchvision import transforms

class MasksDataset(Dataset):
    def __init__(self, transforms, imgs_path, labels_path) -> None:
        super().__init__()
        self.transforms = transforms
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.imgs_list = os.listdir(imgs_path)

    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, idx):
        file_image = 'maksssksksss'+ str(idx) + '.png'
        file_label = 'maksssksksss'+ str(idx) + '.xml'
        img_path = os.path.join(self.imgs_path, file_image)
        label_path = os.path.join(self.labels_path, file_label)

        img = Image.open(img_path).convert('RGB')
        target = generate_target(idx, label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

if __name__ == '__main__':
    imgs_path = './archive/images'
    label_path = './archive/annotations'
    data_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    MDS = MasksDataset(data_transform, imgs_path, label_path)
    dataloader = DataLoader(MDS, batch_size=4, shuffle=False, collate_fn=collate_fn)  ## 使用自定义的collate_fn可以避免图片的大小不一致导致出错的问题
    imgs, targets = next(iter(dataloader))
    print(imgs)
    print()
    print(targets)
