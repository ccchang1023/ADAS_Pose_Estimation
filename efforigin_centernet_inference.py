import os, time, sys
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
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
IMG_WIDTH = 1600
IMG_HEIGHT = 360
MODEL_SCALE = 8

flip_rate = 0.5
flip = False
use_mesh = True
use_bg = True

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

    if use_bg is True:
        bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
        bg = bg[:, :img.shape[1] // 4]
        img = np.concatenate([bg, img, bg], 1)

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
        if use_bg is True:
            y = (y + img.shape[1] // 4) * IMG_WIDTH / (img.shape[1] * 1.5) / MODEL_SCALE
        else:
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

import numpy
numpy.random.bit_generator = numpy.random._bit_generator

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

        ###strong aug
        # iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(0,0.03*255)),
        # iaa.Sometimes(0.5,iaa.GammaContrast((0.2,1.7))),

        ###weak aug
        # iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(0,0.003*255)),
        # iaa.Sometimes(0.5,iaa.GammaContrast((0.6,1.2))),


#         iaa.AddToHueAndSaturation(from_colorspace="RGB",value=(-20, 20))  #Hue-> color, saturation -> saido
    ])
    def __call__(self, img, mask=None):
        img = np.array(img)        
        return self.aug.augment_image(image=img)
#         return self.aug(image=img, segmentation_maps=label)

trans = transforms.Compose([
        ###Color jitter : brightness=0, contrast=0, saturation=0, hue=0
        # transforms.ColorJitter(0.5,0.5,0.5,0.5),  ###strong aug  
        # transforms.ColorJitter(0.1,0.0,0.1,0.1),  ###weak aug  
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
#         self.data = global_car_data
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
#         print(idx)
        idx += self.offset
        idx = self.indices[idx]
        img = cv2.imread('./dataset/masked_train/' + train_pd['ImageId'].iloc[idx] + '.jpg')
        img = np.array(img[:,:,::-1])
        
        if self.is_validate:
            flip = False
        else:
            flip = True if np.random.random()>flip_rate else False

        mask, regr = get_mask_and_regr(img, train_pd['PredictionString'][idx],flip=flip)
        img_pre = preprocess_image(img,flip)  #shape(batch,512,512), #range: [0~1]
        
        img_pil = (img_pre*255).astype('uint8')
        img_pil = Image.fromarray(img_pil)  #ndarray: Take uint8 as input, range[0~255], #imgpil -> (512,512,3), (0~255)
        img_pre_trans = self.transform(img_pil)
        
#         print(np.shape(img))        #(2710, 3384, 3)
#         print(np.shape(img_pre))    #(320, 1024, 3)
#         print(np.shape(mask))       #(40, 128)
#         print(np.shape(regr))       #(40, 128,7)
        
        regr = np.transpose(regr,(2,0,1))
        
#         label = torch.as_tensor(label, dtype=torch.uint8)    #value: 0~9, shape(1)
#         img_pre = torch.as_tensor(img_pre, dtype=torch.float32) #For distribution computing

        return img_pre_trans, mask, regr

    def __len__(self):
        return self.len

    
class TestDataset(Dataset):
    def __init__(self,data_len=None):
        self.transform = trans_test
        self.len = data_len
        
    def __getitem__(self, idx):
        img = cv2.imread('./dataset/masked_test/{}.jpg'.format(test_pd['ImageId'].iloc[idx]))
        
        img = np.array(img[:,:,::-1])
        img_pre = preprocess_image(img)  #shape(batch,512,512), #range: [0~1]
        img_pil = (img_pre*255).astype('uint8')
        img_pil = Image.fromarray(img_pil)  #ndarray: Take uint8 as input, range[0~255], #imgpil -> (512,512,3), (0~255)
        img_pre_trans = self.transform(img_pil)        
        return img_pre_trans   #return img_pre_trans, _ will cause strange behavior (load very slow after 10 batch)
    def __len__(self):
        return self.len


### prediction 1-7: ['pitch_cos', 'pitch_sin', 'roll', 'x', 'y', 'yaw', 'z']
from CenterNet.src.lib.models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, RegLoss_without_ind
f_loss = FocalLoss()
l1_loss = RegLoss_without_ind()

def gaussian_filter(a):
    b = torch.nn.MaxPool2d(3,stride=1,padding=1)(a)
    eq = torch.eq(a,b).float()
    a = a*eq
    return a

### prediction 1-7: ['pitch_cos', 'pitch_sin', 'roll', 'x', 'y', 'yaw', 'z']
from CenterNet.src.lib.models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, RegLoss_without_ind
f_loss = FocalLoss()
l1_loss = RegLoss_without_ind()

def gaussian_filter(a):
    b = torch.nn.MaxPool2d(3,stride=1,padding=1)(a)
    eq = torch.eq(a,b).float()
    a = a*eq
    return a

def criterion_new(pred, mask, regr, weight=0.5, result_average=True, multiloss=False):
    #pred:(batch,8,64,64) mask:(batch,64,64) regr:(batch,7,64,64)
    scalar = 1

    ###Peak    
    # pred_mask = torch.sigmoid(pred[:, 0])
    pred_mask = torch.clamp(torch.sigmoid(pred[:, 0]), min=1e-12, max=1-1e-12) #(batch,64,64)
    # pred_mask = torch.clamp(torch.sigmoid(pred[:, 0]), min=1e-6, max=1-1e-6) #(batch,64,64)
    pred_mask = pred_mask.unsqueeze(1)
    gaussian_pred = gaussian_filter(pred_mask)
    gaussian_pred = pred_mask
    mask_loss = scalar*f_loss(gaussian_pred, mask.unsqueeze(1))
    
    ###No focal
    # pred_mask = torch.sigmoid(pred[:, 0])
    # mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    # mask_loss = -mask_loss.mean(0).sum()    

#     print(regr.size()) #(batch,7,64,64)
#     print(pred.size()) #(batch,8,64,64), pred[:,1] ->(batch,64,64) ; pred[:,1:3] ->(batch,2,64,64)
    ### task A 训练的收敛后，再把 A 和 B join 到一起训练
    if multiloss:
        pred_xyz = torch.cat((pred[:,4].unsqueeze(1),pred[:,5].unsqueeze(1),pred[:,7].unsqueeze(1)), 1)
        pred_rollyaw = torch.cat((pred[:,3].unsqueeze(1),pred[:,6].unsqueeze(1)), 1)
        pred_pitch = torch.cat((pred[:,1].unsqueeze(1),pred[:,2].unsqueeze(1)), 1)
        
        gt_xyz = torch.cat((regr[:,3].unsqueeze(1),regr[:,4].unsqueeze(1),regr[:,6].unsqueeze(1)), 1)
        gt_rollyaw = torch.cat((regr[:,2].unsqueeze(1),regr[:,5].unsqueeze(1)), 1)
        gt_pitch = torch.cat((regr[:,0].unsqueeze(1),regr[:,1].unsqueeze(1)), 1)
        
        xyz_loss = scalar*l1_loss(pred_xyz,gt_xyz,mask)
        rollyaw_loss = scalar*l1_loss(pred_rollyaw,gt_rollyaw,mask)
        pitch_loss = scalar*l1_loss(pred_pitch,gt_pitch,mask)
        loss = mask_loss + xyz_loss + rollyaw_loss + pitch_loss
    else:
        regr_loss = mask_loss
        loss = mask_loss

    if not result_average:
        loss *= pred.shape[0]
        
    return loss ,mask_loss , xyz_loss, rollyaw_loss, pitch_loss


from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
class double_conv(nn.Module):
    '''(conv => ReLU => BN) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh

class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b4')
        self.base_model = EfficientNet.from_pretrained('efficientnet-b5')
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b7')
        
        ###Revised model
        if use_mesh == True:
            self.conv0 = double_conv(5, 128)
        else:
            self.conv0 = double_conv(3, 128)
        self.conv1 = double_conv(128, 256)
        self.conv2 = double_conv(256, 512)
        self.conv3 = double_conv(512, 1024)

        ###Origin model
        # self.conv0 = double_conv(5, 64)
        # self.conv1 = double_conv(64, 128)
        # self.conv2 = double_conv(128, 512)
        # self.conv3 = double_conv(512, 1024)
        
        self.mp = nn.MaxPool2d(2)
        self.dp = nn.Dropout(0.5)
        
        ###eff-b0
        # self.up1 = up(1282 + 1024, 512)
        # self.up2 = up(512 + 512, 256)
        
        ###eff-b4
        # self.up1 = up(1794 + 1024, 512)
        # self.up2 = up(512 + 512, 256)

        ###eff-b5
        if use_mesh == True:
            self.up1 = up(2050 + 1024, 512)
        else:
            self.up1 = up(2048 + 1024, 512)
        self.up2 = up(512 + 512, 256)  

        ###eff-b7
        # if use_mesh == True:
        #     self.up1 = up(2562 + 1024, 512)
        # else:
        #     self.up1 = up(2560 + 1024, 512)
        # self.up2 = up(512 + 512, 256)
        
        
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        if use_mesh == True:
            mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
            x0 = torch.cat([x, mesh1], 1)
        else:
            x0 = x

        ###Revised model
        x1 = self.dp(self.mp(self.conv0(x0)))
        x2 = self.dp(self.mp(self.conv1(x1)))
        x3 = self.dp(self.mp(self.conv2(x2)))
        x4 = self.dp(self.mp(self.conv3(x3)))

        ###Origin model
        # x1 = self.mp(self.conv0(x0))
        # x2 = self.mp(self.conv1(x1))
        # x3 = self.mp(self.conv2(x2))
        # x4 = self.mp(self.conv3(x3))
        
#         x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        x_center = x[:, :, :, :]
        
        feats = self.base_model.extract_features(x_center)    #[1, 1280, 16, 24] for eff-b0
#         print("eff extract size:",feats.size())
        
        ### Pad with zero (don't know why)
        # bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        # print("bg size:",bg.size())
        # feats = torch.cat([bg, feats, bg], 3)
        # print("feats size:",feats.size())
        
        # Add positional info
        if use_mesh == True:
            mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
            feats = torch.cat([feats, mesh2], 1)
        
        x = self.up1(feats, x4)
#         print("up1 size:",x.size())
        x = self.up2(x, x3)
#         print("up2 size:",x.size())
        x = self.outc(x)
        return x
    




img = imread("./dataset/masked_train/{}.jpg".format(train_pd['ImageId'].iloc[0]))
IMG_SHAPE = img.shape
DISTANCE_THRESH_CLEAR = 2.5


def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords
points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train_pd['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr
print('len(points_df)', len(points_df))
points_df.head(1)

xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)
print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))

print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*xzy_slope.coef_))


def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        if use_bg is True:
            y = (y + IMG_SHAPE[1] // 4) * IMG_WIDTH / (IMG_SHAPE[1] * 1.5) / MODEL_SCALE
        else:
            y = y * IMG_WIDTH / IMG_SHAPE[1] / MODEL_SCALE
        return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

def extract_coords(prediction, flipped=False, logits_th=None, gt=False, peak_output=None):
    ###prediction -> (batch,h,w)
    logits = prediction[0]
    regr_output = prediction[1:]
    if gt:
        points = np.argwhere(logits > 0)
        
    else:
        points = np.argwhere(peak_output > 0)
#         points = np.argwhere(peak_output[0]>logits_th)
#         points = np.argwhere(logits > logits_th)
        
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] =\
                optimize_xy(r, c,
                            coords[-1]['x'],
                            coords[-1]['y'],
                            coords[-1]['z'], flipped=False)
    coords = clear_duplicates(coords)
    return coords

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


if __name__=='__main__':

    cv_fold = [0,1,2,3,4]
    batch_size = 6
    # model_root = "./cv_models_0118/"
    model_root = "./tmp_0117_18/"
    ensemble_models = []
    for file_name in os.listdir(model_root):
        # if file_name.find("1600x360")==-1:
        #     continue
        model = MyUNet(8)
        model.cuda()
        # model.load_state_dict(torch.load("./cv_models_0118/{}".format(file_name)))
        model.load_state_dict(torch.load("{}/{}".format(model_root,file_name)))
        ensemble_models.append(model)

    model_num = len(ensemble_models)

    predictions = []
    test_pd = pd.read_csv("./dataset/sample_submission.csv")
    indices_len = len(test_pd)
    print("indices_len",indices_len)
    test_dataset = TestDataset(data_len=indices_len)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            
    k = 18
    with torch.no_grad():
        for img in tqdm(test_loader):
            b_num = img.size(0)
            img = img.to(device)

            ###Average Ensemble
            pred_list = torch.Tensor([]).to(device)
            for i in range(model_num):
                pred = ensemble_models[i](img) #(batch,8,h,w)
                # print("size1",pred.size())
                h,w = pred.size(2),pred.size(3)
                pred = pred.view(b_num,8,-1) #(batch,8,hxw)
                # print("size2",pred.size())
                pred_list = torch.cat((pred_list,pred.unsqueeze(3)),dim=3) #pred_list: (batch_num,8,hxw,model_num)
            output_tensor = torch.mean(pred_list,dim=3).view(b_num,8,h,w) #output_tensor (batch_num,8,h,w)

            output = output_tensor.cpu().numpy()
            peak_output = torch.sigmoid(output_tensor[:,0])
            peak_output = (peak_output.float())
            score, inds = torch.topk(peak_output.view(b_num,-1),k=k)  #score (8,40)
            minScore, indc = score.min(dim=1)
            minScore = minScore.unsqueeze(1)           #minScore (8,1)
#             print(np.shape(minScore))
            h,w = peak_output.size(1), peak_output.size(2)
            peak_output = peak_output.view(b_num,-1)
            peak_output = torch.where(peak_output>=minScore,peak_output,torch.zeros_like(peak_output))
            peak_output = peak_output.view(b_num,h,w)
            peak_output = peak_output.cpu().numpy()            
            for idx,out in enumerate(output):     #out -> (8,h,w)
                coords = extract_coords(out,logits_th=0,gt=False,peak_output=peak_output[idx])
                s = coords2str(coords)
                predictions.append(s)    
    
    print(np.shape(predictions))
    test = pd.read_csv(PATH + 'sample_submission.csv')
    test['PredictionString']= predictions
    file_name = "final_pred_fold_1718_7folds.csv"
    test.to_csv(file_name, index=False)
    test.head()        





