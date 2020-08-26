'''
Scout utility functions.
'''

import os, random
import numpy as np
import torch

__dir__ = os.path.dirname(os.path.realpath(__file__))
__models__ = os.path.join(__dir__, "models")
__configs__ = os.path.join(__models__, "configs")
default_config = os.path.join(__configs__, "dna_r941_21bp.toml")


def init(seed, device):
    """
    Initialise random libs and setup cudnn 

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if device == "cpu": return 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    assert(torch.cuda.is_available())


class ChunkData:
    def __init__(self, blocks, targets):
        self.blocks = blocks.reshape((*blocks.shape[:-2],-1)).transpose((0,2,1))
        self.targets = np.expand_dims(targets, (1,2))
    def __getitem__(self, i):
        return (self.blocks[i], self.targets[i])
    def __len__(self):
        return len(self.blocks)



# def load_model(dirname, device, weights=None, half=False, chunksize=0, use_rt=False):
#     """
#     Load a model from disk
#     """
#     if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
#         dirname = os.path.join(__models__, dirname)

#     if not weights: # take the latest checkpoint
#         weight_files = glob(os.path.join(dirname, "weights_*.tar"))
#         if not weight_files:
#             raise FileNotFoundError("no model weights found in '%s'" % dirname)
#         weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])

#     device = torch.device(device)
#     config = os.path.join(dirname, 'config.toml')
#     weights = os.path.join(dirname, 'weights_%s.tar' % weights)
#     model = Model(toml.load(config))

#     state_dict = torch.load(weights, map_location=device)
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k.replace('module.', '')
#         new_state_dict[name] = v

#     model.load_state_dict(new_state_dict)

#     if use_rt:
#         model = CuModel(model.config, chunksize, new_state_dict)

#     if half: model = model.half()
#     model.eval()
#     model.to(device)
#     return model
