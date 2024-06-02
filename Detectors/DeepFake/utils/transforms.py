from torchvision import transforms
import numpy as np


def transform_handler(train=False, image_size=224, crop=False, jitter=False, noise=False):
    assert (type(crop) is bool) or (type(jitter) is bool), "crop과 jitter의 value type는 bool만 가능합니다. [True or False]"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        if crop:
            img_trans = [transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))]
        else:
            img_trans = [transforms.Resize((image_size, image_size))]

        if jitter:
            img_trans.append(transforms.ColorJitter(brightness=(0.5, 0.9),
                                                    contrast=(0.4, 0.8),
                                                    saturation=(0.7, 0.9),
                                                    hue=(-0.2, 0.2)))
        if noise:
            img_trans.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))

        img_trans += [transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean, std)]

        img_trans = transforms.Compose(img_trans)

    else:
        img_trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    print('-------------------[Loaded transform]')
    return img_trans
