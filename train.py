from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import torch
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Cfg.load_config_from_name('vgg_seq2seq')


parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, default="vovnet")
parser.add_argument("--total_step", type=int, default= 40000)
parser.add_argument("--step", type=int, default= 10)
parser.add_argument("--valid", type=int, default= 500)
parser.add_argument("--metrics", type=int, default= 300)
parser.add_argument("--batch_sie", type=int, default= 8)
parser.add_argument("--root", type=str, default="../data")
parser.add_argument("--dataset_name", type=str, default="ocr")

args = parser.parse_args()

def main():

    dataset_params = {
        'name':args["dataset_name"], 
        'data_root':args["root"], 
        'train_annotation':'train.txt', 
        'valid_annotation':'val.txt', 
        'image_max_width':1920,
        'image_height': 32
    }

    params = {
            'print_every': args["step"], 
            'valid_every': args["valid"], 
            'iters': args["total_step"], 
            'export':'weights/vietocrvovnet.pth', 
            'metrics': args["metrics"],
            'batch_size': args["batch_size"],
            }


    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = device
    #config['backbone'] = "vovnet"


    trainer = Trainer(config, pretrained=False)

    print(trainer.model)

    trainer.train()
