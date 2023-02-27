from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

config = Cfg.load_config_from_name('vgg_seq2seq')

dataset_params = {
    'name':'Hands-VNOnDB', 
    'data_root':'../data', 
    'train_annotation':'train.txt', 
    'valid_annotation':'val.txt', 
    'image_max_width':1920,
    'image_height': 32
}

params = {
         'print_every': 1, 
         'valid_every': 1, 
          'iters': 2, 
          'export':'weights/vietocrvovnet.pth', 
          'metrics': 200,
          'batch_size': 32,
         }


config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = device
config['backbone'] = "vovnet"


trainer = Trainer(config, pretrained=False)

print(trainer.model)

trainer.train()