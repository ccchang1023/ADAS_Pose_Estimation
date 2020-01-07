import os, time, sys
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# import matplotlib.pyplot as plt

# from torchsummary import summary
# import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from math import sin, cos
# from tqdm import tqdm

device = "cuda"
SWITCH_LOSS_EPOCH = 5
IMG_WIDTH = 512
IMG_HEIGHT = 512
MODEL_SCALE = 8

PATH = "./dataset/"
train_pd = pd.read_csv(PATH + 'train_remove.csv')
test_pd = pd.read_csv(PATH + 'sample_submission.csv')
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def get_img_coords(s):
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

def rotate(x, angle): 
    x = x + angle 
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi 
    return x

def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
#     bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
#     bg = bg[:, :img.shape[1] // 6]
#     img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
#         y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE
        y = (y) * IMG_WIDTH / (img.shape[1]) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr


from imgaug import augmenters as iaa
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
#         iaa.Scale((640, 480)),
#         iaa.Fliplr(0.5),
#         iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.75))),
#         iaa.Sometimes(0.1, iaa.AverageBlur(1.2)),
#         iaa.Sometimes(1, iaa.Affine(rotate=(-20, 20),order=[0, 1],translate_px={"x":(-2, 2),"y":(-2,2)},mode='symmetric')),
#         iaa.Sometimes(0.2,iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25))),
#         iaa.Sometimes(0.1, iaa.SaltAndPepper(0.05,False)),
#         iaa.Invert(0.5),
#         iaa.Add((-5, 5)), # change brightness of images (by -10 to 10 of original value)
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(0,0.01*255)),
        iaa.Sometimes(0.5,iaa.GammaContrast((0.3,1.5))),
#         iaa.AddToHueAndSaturation(from_colorspace="RGB",value=(-20, 20))  #Hue-> color, saturation -> saido
    ])
    def __call__(self, img, mask=None):
        img = np.array(img)        
        return self.aug.augment_image(image=img)
#         return self.aug(image=img, segmentation_maps=label)

trans = transforms.Compose([
#         transforms.ColorJitter(0.,0.2,0.,0.),
        ImgAugTransform(),
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2732385,0.28948802,0.31470126],std=[0.19721317,0.20766443,0.20611505])
    ])

trans_val = transforms.Compose([
        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1
        transforms.Normalize(mean=[0.2732385,0.28948802,0.31470126],std=[0.19721317,0.20766443,0.20611505])
    ])

trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.26370072,0.28066522,0.30648127],std=[0.19954063,0.20964707,0.2084653])
    ])

class ADDataset(Dataset):
    def __init__(self,data_len=None, is_validate=False,validate_rate=None,indices=None):
        self.is_validate = is_validate
        if data_len == None:
            data_len = len(self.data)
        
        self.indices = indices
        if self.is_validate:
            self.len = int(np.ceil(data_len*validate_rate))
            self.offset = int(data_len*(1-validate_rate))
            self.transform = trans_val
        else:
            self.len = int(data_len*(1-validate_rate))
            self.offset = 0
            self.transform = trans
        
    def __getitem__(self, idx):
        idx += self.offset
        idx = self.indices[idx]
        img = cv2.imread('./dataset/masked_train/' + train_pd['ImageId'].iloc[idx] + '.jpg')
        img = np.array(img[:,:,::-1])

        mask, regr = get_mask_and_regr(img, train_pd['PredictionString'][idx])
        img_pre = preprocess_image(img)  #shape(batch,512,512), #range: [0~1]
        
        img_pil = (img_pre*255).astype('uint8')
        img_pil = Image.fromarray(img_pil)  #ndarray: Take uint8 as input, range[0~255], #imgpil -> (512,512,3), (0~255)
        img_pre_trans = self.transform(img_pil)
        regr = np.transpose(regr,(2,0,1))
        return img_pre_trans, mask, regr

    def __len__(self):
        return self.len

    
class TestDataset(Dataset):
    def __init__(self,data_len=None):
        self.transform = trans_test
        self.len = data_len
        
    def __getitem__(self, idx):
        img = cv_read('./dataset/masked_test/{}.jpg'.format(test_pd['ImageId'].iloc[idx]))
        
        img = np.array(img[:,:,::-1])
        img_pre = preprocess_image(img)  #shape(batch,512,512), #range: [0~1]
        img_pil = (img_pre*255).astype('uint8')
        img_pil = Image.fromarray(img_pil)  #ndarray: Take uint8 as input, range[0~255], #imgpil -> (512,512,3), (0~255)
        img_pre_trans = self.transform(img_pil)        
        return img_pre_trans   #return img_pre_trans, _ will cause strange behavior (load very slow after 10 batch)
    def __len__(self):
        return self.len


from CenterNet.src.lib.models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, RegLoss_without_ind
f_loss = FocalLoss()
l1_loss = RegLoss_without_ind()

def criterion_new(prediction, mask, regr, weight=0.5, result_average=True):
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = f_loss(pred_mask, mask)

    # pred_regr = prediction[:, 1:]
    # regr_loss = l1_loss(pred_regr,regr,mask)
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = weight*mask_loss +(1-weight)* regr_loss
    if not result_average:
        loss *= prediction.shape[0]
    return loss ,mask_loss , regr_loss


def criterion(prediction, mask, regr,weight=0.4, result_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
  
    # Sum
    loss = weight*mask_loss +(1-weight)* regr_loss
    if not result_average:
        loss *= prediction.shape[0]
    return loss ,mask_loss , regr_loss



if __name__=='__main__':
    vr = 0.1
    batch_size = 6
    num_workers = 12
    train_pd = pd.read_csv("./dataset/train_remove.csv")
    indices_len = len(train_pd)
    print("len of train indices:",indices_len)
    indices = np.arange(indices_len)
    train_dataset = ADDataset(data_len=indices_len,is_validate=False,validate_rate=vr,indices=indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = ADDataset(data_len=indices_len,is_validate=True,validate_rate=vr,indices=indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    from light_seg_model import get_segmentation_model
    # model = get_segmentation_model("efficientnet", dataset=train_dataset,aux=False, norm_layer=torch.nn.BatchNorm2d).to(device)
    model = get_segmentation_model("efficientnet_b7", dataset=train_dataset,aux=False, output_size=(64,64), norm_layer=torch.nn.BatchNorm2d).to(device)

    if device=='cuda':
        model.cuda()

    lr = 1e-4
    lr_period = 5
    epochs = 50
    val_freq = 1
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2,factor=0.2)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs, 10) * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=lr_period,T_mult=1,eta_min=1e-6) #original 

    min_loss = 100000
    best_model_dict = None

    for ep in range(1,epochs+1):
        model.train()
        print("Epoc:",ep)
        for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(train_loader):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            optimizer.zero_grad()
            output = model(img_batch)
            loss, mask_loss , regr_loss = criterion_new(output, mask_batch, regr_batch)
            loss.backward()
            optimizer.step()
        print('Epoch:{} lr:{:.5f} loss:{:.6f} mloss:{:.6f} rloss{:.6f}'.
               format(ep,optimizer.param_groups[0]['lr'],loss.data,mask_loss.data,regr_loss.data))
    #     lr_scheduler.step()

        if ep%val_freq==0:
            model.eval()
            val_loss = 0
            data_num = 0
            with torch.no_grad():
                for img_batch, mask_batch, regr_batch in val_loader:
                    img_batch = img_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    regr_batch = regr_batch.to(device)
                    output = model(img_batch)
                    val_loss += criterion_new(output, mask_batch, regr_batch, result_average=False)[0].data
                    img_batch.size()
                    data_num += img_batch.size(0)
                    
            val_loss /= data_num
            print('Val loss: {:.5f}'.format(val_loss))
            
            lr_scheduler.step(val_loss)
            
            if val_loss < min_loss:
                min_loss = val_loss
                best_model_dict = model.state_dict()
                print("./saved_model/Ep:{}_loss{:.4f}".format(ep,min_loss))
                torch.save(best_model_dict, "./saved_model/effseg_512x512_halfloss_Ep{}_loss{:.4f}".format(ep,min_loss))