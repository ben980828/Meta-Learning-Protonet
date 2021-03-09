import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
class ProNet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super(ProNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

    def forward(self, input_data):
        x = self.encoder(input_data)
        return x.view(x.size(0), -1)

def euclidean_dist(x, y):
    n = x.shape[0]
    m = y.shape[0]
    a = x.unsqueeze(1).expand(n, m, -1)
    b = y.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)

    return logits

def predict(model, data_loader):
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):

            # split data into support and query data
            support_input = data[:5 * 1,:,:,:] 
            query_input   = data[5 * 1:,:,:,:]
            support_input = support_input.cuda()
            query_input = query_input.cuda()

            label_encoder = {target[i * 1] : i for i in range(5)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[5 * 1:]])

            model_output = model(support_input)
            output = model_output.reshape(1, 5, -1).mean(dim=0)# means of classes along n_support shots

            logits = euclidean_dist(model(query_input), output)
            pred = torch.argmax(logits, dim=1).tolist()
            pred.insert(0, i)
            prediction_results.append(pred)

    return prediction_results

if __name__=='__main__':
    pyfile = sys.argv[0]
    test_csv = sys.argv[1]
    test_data_dir = sys.argv[2]
    testcase_csv = sys.argv[3]
    output_csv = sys.argv[4]

    test_dataset = MiniDataset(test_csv, test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=5 * (15 + 1),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(testcase_csv))
    

    model = ProNet()
    state = torch.load('./pronet_eucdis_ep184_acc0.5160000118613243±0.01767819107550266.pth')
    model.load_state_dict(state)
    model = model.cuda()
    model.eval()
    prediction_results = predict(model, test_loader)

    episode_label = []
    episode_label.append('episode_id')
    for i in range(15*5):
        episode_label.append('query{}'.format(i))

    prediction_results.insert(0, episode_label)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in prediction_results:
            writer.writerow(row)