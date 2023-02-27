import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv3x3(in_channels, out_channels, module_name, posfix, stride = 1, groups = 1, padding = 1):
  return \
  [
      (
          f'{module_name}_{posfix}/conv3x3',
          nn.Conv2d
          (
              in_channels = in_channels, 
              out_channels= out_channels,
              kernel_size= 3,
              stride = stride,
              groups = groups,
              padding = padding
          )
      ),
      (
          f'{module_name}_{posfix}/batchnorm',
          nn.BatchNorm2d(out_channels)
      ),
      (
          f'{module_name}_{posfix}/ReLU',
          nn.ReLU(inplace=True)
      )
  ]

def conv1x1(in_channels, out_channels, module_name, posfix, stride = 1, groups = 1, padding = 0):
  return \
  [
      (
          f'{module_name}_{posfix}/conv1x1',
          nn.Conv2d
          (
              in_channels = in_channels,
              out_channels = out_channels,
              kernel_size= 1,
              stride = stride,
              groups = groups,
              padding = padding
          )
      ),
      (
          f'{module_name}_{posfix}/batchnorm',
          nn.BatchNorm2d(out_channels)
      ),
      (
          f'{module_name}_{posfix}/ReLU',
          nn.ReLU(inplace = True)
      )
  ]

class OSAModule(nn.Module):
    def __init__(self, in_channels, stage_channels, out_channels, num_layers, module_name, identity_mapping = False):
        super(OSAModule, self).__init__()

        self.identity_mapping = identity_mapping

        self.layers = nn.ModuleList()

        input_ch = in_channels
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    OrderedDict(
                        conv3x3(
                            in_channels = input_ch,
                            out_channels = stage_channels,
                            module_name = module_name,
                            posfix = i
                        )
                    )
                )
            )
            input_ch = stage_channels

        out_stage_ch = in_channels + num_layers * stage_channels

        self.feat_agg = nn.Sequential(
            OrderedDict(
                conv1x1(
                    in_channels = out_stage_ch,
                    out_channels = out_channels,
                    module_name = module_name,
                    posfix = "agg"
                )
            )
        )
  
    def forward(self, x):

        if self.identity_mapping:
            x_idx = x

        outputs = [x]
        
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        
        x_cat = torch.cat(outputs, dim = 1)
        x_agg = self.feat_agg(x_cat)
        if self.identity_mapping:
            x_agg = x_agg + x_idx
        return x_agg

class OSAStage(nn.Sequential):
    def __init__(self, in_channels, stage_channels, out_channels, num_osa_block, num_conv_per_osa, stage_num):
        super(OSAStage, self).__init__()

        if stage_num != 2 and stage_num != 5:
        
            self.add_module(
                    'Pooling',
                    nn.AvgPool2d(kernel_size= [2, 1], stride=[2, 1], padding=0)
                )

        if stage_num != 5:
        
            self.add_module(f'OSA{stage_num}_1',
                OSAModule(in_channels,
                            stage_channels,
                            out_channels,
                            num_conv_per_osa,
                            f'OSA{stage_num}_1'))
            
            for i in range(num_osa_block - 1):
                self.add_module(f'OSA{stage_num}_{i+2}',
                    OSAModule(out_channels,
                                stage_channels,
                                out_channels,
                                num_conv_per_osa,
                                f'OSA{stage_num}_{i+2}',
                                identity_mapping = True))
        elif stage_num == 5:
            if num_osa_block == 1:
                self.add_module(f'OSA{stage_num}_1',
                    OSAModule(in_channels,
                                stage_channels,
                                256,
                                num_conv_per_osa,
                                f'OSA{stage_num}_1'))
            else:
                self.add_module(f'OSA{stage_num}_1',
                OSAModule(in_channels,
                            stage_channels,
                            out_channels,
                            num_conv_per_osa,
                            f'OSA{stage_num}_1'))
            
                for i in range(num_osa_block - 1):
                    if i == num_osa_block - 2:
                        self.add_module(f'OSA{stage_num}_{i+2}',
                            OSAModule(out_channels,
                                        stage_channels,
                                        256,
                                        num_conv_per_osa,
                                        f'OSA{stage_num}_{i+2}',
                                        identity_mapping = False))
                    else:
                        self.add_module(f'OSA{stage_num}_{i+2}',
                            OSAModule(out_channels,
                                        stage_channels,
                                        out_channels,
                                        num_conv_per_osa,
                                        f'OSA{stage_num}_{i+2}',
                                        identity_mapping = True))


class OSAConfig:
  def __init__(self, name):
    self._get_config_from_name(name)
  
  def _get_config_from_name(self, name):
    if name.lower() == "vovnet-39":
      self.stage_ch = [128, 160, 192, 224]
      self.agg_ch = [256, 512, 768, 1024]
      self.osa_per_stage = [1, 1, 2, 2]
      self.num_conv_in_osa = 5
    elif name.lower() == "vovnet-57":
      self.stage_ch = [128, 160, 192, 224]
      self.agg_ch = [256, 512, 768, 1024]
      self.osa_per_stage = [1, 1, 4, 3]
      self.num_conv_in_osa = 5
    elif name.lower() == "vovnet-27-slim":
      self.stage_ch = [64, 80, 96, 112]
      self.agg_ch = [128, 256, 384, 512]
      self.osa_per_stage = [1, 1, 1, 1]
      self.num_conv_in_osa = 5
    else:
      print("name not valid, name must be: vovnet-39 | vovnet-57 | vovnet-27-slim")
      raise Exception()
  

class VoVNet(nn.Module):
    def __init__(self, name):
        super(VoVNet, self).__init__()
        self.osaConfig = OSAConfig(name)
        # stem
        stem = conv3x3(
            in_channels = 3,
            out_channels = 64,
            module_name = "stem",
            posfix = "1",
            stride = 2
        )

        stem += conv3x3(
            in_channels = 64,
            out_channels = 64,
            module_name = "stem",
            posfix = "2",
            stride = 1
        )

        stem += conv3x3(
            in_channels = 64,
            out_channels = 128,
            module_name = "stem",
            posfix = "3",
            stride = 2
        )

        in_ch = [128] + self.osaConfig.agg_ch[:-1]

        self.add_module("stem", nn.Sequential(OrderedDict(stem)))

        # Config OSA
        
        self.stage_names = []
        for i in range(4): #num_stages
            name = 'stage%d' % (i+2)
            self.stage_names.append(name)
            self.add_module(name,
                            OSAStage(in_ch[i],
                                        self.osaConfig.stage_ch[i],
                                        self.osaConfig.agg_ch[i],
                                        self.osaConfig.osa_per_stage[i],
                                        self.osaConfig.num_conv_in_osa,
                                        i+2))
        #from source code
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        # print("stem: ", x.size())
        for name in self.stage_names:
            x = getattr(self, name)(x)
            # print("osa: ", x.size())
        return x

class Identity(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        return x

class VoVNetBackbone(nn.Module):
    def __init__(self):
        super(VoVNetBackbone, self).__init__()
        self.feature = VoVNet("vovnet-27-slim")
    
    def forward(self, x):
        # print("input size: ", x.size())
        conv = self.feature(x)
        
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)

        # print("output size: ", conv.size())
        return conv
