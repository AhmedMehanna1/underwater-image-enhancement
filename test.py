import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
# my import
from model import AIMnet
from dataset_all import TestData

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

bz = 1
#model_root = 'pretrained/model.pth'
model_root = 'model/ckpt/model_best_student.pth'
input_root = 'test/u45'
save_path = 'test/u45/enhanced'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root, 256)
data_load = data.DataLoader(Mydata_, batch_size=bz)

model = AIMnet().cuda()
model = nn.DataParallel(model, device_ids=[0, 1])
model.load_state_dict(checkpoint)
model.eval()
print('START!')
if 1:
    print('Load model successfully!')
    for data_idx, data_ in enumerate(data_load):
        data_input, data_la = data_

        data_input = Variable(data_input).cuda()
        data_la = Variable(data_la).cuda()
        with torch.no_grad():
            result, _ = model(data_input, data_la)
            name = Mydata_.A_paths[data_idx].split('/')[-1]
            print(name)
            temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            temp_res = (temp_res*255).astype(np.uint8)
            temp_res = Image.fromarray(temp_res)
            temp_res.save('%s/%s' % (save_path, name))
            print('result saved!')

print('finished!')
