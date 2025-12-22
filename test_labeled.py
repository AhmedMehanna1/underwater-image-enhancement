import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dataset_all import make_dataset
# my import
from model import AIMnet
from utils import AverageMeter, compute_psnr_ssim, to_psnr

class TestLabeledData(data.Dataset):
    def __init__(self, dataroot, fineSize):
        super().__init__()
        self.root = dataroot
        self.fineSize = fineSize

        self.dir_A = os.path.join(self.root, 'input')
        self.dir_B = os.path.join(self.root, 'GT')
        self.dir_C = os.path.join(self.root, 'LA')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index]).convert("RGB")

        resized_a = A.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        resized_b = B.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        resized_c = C.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)

        # transform to (0, 1)
        tensor_a = self.transform(resized_a)
        tensor_b = self.transform(resized_b)
        tensor_c = self.transform(resized_c)

        return tensor_a, tensor_b, tensor_c

    def __len__(self):
        return len(self.A_paths)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

bz = 1
#model_root = 'pretrained/model.pth'
model_root = 'model/ckpt/model_best_student.pth'
input_root = 'test/uieb'
save_path = 'test/uieb/enhanced'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)
Mydata_ = TestLabeledData(input_root, 256)
data_load = data.DataLoader(Mydata_, batch_size=bz)
test_psnr = AverageMeter()
test_ssim = AverageMeter()
psnr_val = []

model = AIMnet().cuda()
model = nn.DataParallel(model, device_ids=[0, 1])
model.load_state_dict(torch.load(model_root))
#model.load_state_dict(checkpoint['state_dict'])
model.eval()
print('START!')
if 1:
    print('Load model successfully!')
    tbar = tqdm(data_load, ncols=130)
    data_load = iter(data_load)
    tbar = range((len(data_load)))
    tbar = tqdm(tbar, ncols=130, leave=True)
    for data_idx in tbar:
        data_ = next(data_load)
        data_input, data_label, data_la = data_
        name = Mydata_.A_paths[data_idx].split('/')[-1]

        data_input = Variable(data_input).cuda()
        data_label = Variable(data_label).cuda()
        data_la = Variable(data_la).cuda()
        with torch.no_grad():
            result, _ = model(data_input, data_la)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(result, data_label)
            test_psnr.update(temp_psnr, N)
            test_ssim.update(temp_ssim, N)
            psnr_val.extend(to_psnr(result, data_label))

            tbar.set_description('Test image: {} | PSNR: {:.4f}, SSIM: {:.4f}|'.format(name, test_psnr.avg, test_ssim.avg))

            temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            temp_res = (temp_res*255).astype(np.uint8)
            temp_res = Image.fromarray(temp_res)
            temp_res.save('%s/%s' % (save_path, name))
            #print('result saved!')

print('finished!')
