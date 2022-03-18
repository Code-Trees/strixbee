import warnings
warnings.filterwarnings('ignore')

import gc
import torch
import os

def clean_gpu():
    """Considering we have GPUs"""

    gc.collect()
    torch.cuda.empty_cache()
    return None
    
def gpu_info():
    gpu_info = os.system("nvidia-smi")
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info)