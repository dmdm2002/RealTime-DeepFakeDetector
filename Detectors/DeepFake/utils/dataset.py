from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import PIL.Image as Image
import glob
from Detectors.DeepFake.utils.transforms import transform_handler


class CustomDataset(Dataset):
    def __init__(self, fake_data_path, live_data_path, trans):
        super().__init__()
        fake_images = glob.glob(f'{fake_data_path}/*')
        live_images = glob.glob(f'{live_data_path}/*')

        self.images = []
        for i in range(len(fake_images)):
            self.images.append([fake_images[i], 0])

        for i in range(len(live_images)):
            self.images.append([live_images[i], 1])

        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trans(Image.open(self.images[idx][0]))
        label = self.images[idx][1]

        return image, label


def get_loader(train,
               image_size=224,
               crop=False,
               jitter=False,
               noise=False,
               batch_size=32,
               fake_path=None,
               live_path=None
               ):

    if train:
        trans = transform_handler(train=train,
                                      image_size=image_size,
                                      crop=crop,
                                      jitter=jitter,
                                      noise=noise)

        dataset = CustomDataset(fake_path, live_path, trans)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    else:
        trans = transform_handler(train=train,
                                  image_size=image_size)
        dataset = CustomDataset(fake_path, live_path, trans)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return loader