import numpy
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class DeepFakeInference:
    def __init__(self):
        self.model = torch.load('D:/Side/CodeGate/backup/DeepFake/RepVGG/try_3/ckp/best_13.pt', map_location='cpu').eval()
        self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    def run(self, x: numpy.ndarray):
        x = self.transform(x)
        x = x.unsqueeze(0)
        output = self.model(x)
        output = nn.functional.softmax(output)

        return output.detach().numpy()[0][1]
