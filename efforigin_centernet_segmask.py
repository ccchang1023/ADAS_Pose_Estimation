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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from math import sin, cos
# from mAp import calculate_mAp, get_mAp

device = "cuda"
IMG_WIDTH = 2048
IMG_HEIGHT = 360
MODEL_SCALE = 8

SEG_W = 256
SEG_H = 45

flip_rate = 0.5
flip = False
use_mesh = True
use_bg = True
bg_ratio = 4

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
        # bg = bg[:, :img.shape[1] // 4]
        bg = bg[:, :img.shape[1] // bg_ratio]

        img = np.concatenate([bg, img, bg], 1)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def preprocess_mask_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    if use_bg is True:
        bg = np.zeros_like(img).astype(img.dtype)
        bg = bg[:, :img.shape[1] // 4]
        img = np.concatenate([bg, img, bg], 1)
        
    img = cv2.resize(img, (SEG_W, SEG_H))
    if flip:
        img = img[:,::-1]
    return img

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
            # y = (y + img.shape[1] // 4) * IMG_WIDTH / (img.shape[1] * 1.5) / MODEL_SCALE
            y = (y + img.shape[1] // 4) * IMG_WIDTH / (img.shape[1] * (bg_ratio+2)/bg_ratio) / MODEL_SCALE
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


from imgaug import augmenters as iaa
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
        ###strong aug
        # iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(0,0.03*255)),
        # iaa.Sometimes(0.5,iaa.GammaContrast((0.2,1.7))),
        ###weak aug
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(0,0.01*255)),
        iaa.Sometimes(0.5,iaa.GammaContrast((0.6,1.2))),
        ])

# class ImgAugTransform: ###Strong aug 2
#     def __init__(self):
#         self.aug = iaa.Sequential([
#         iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.75))),
# #         iaa.Sometimes(0.5, iaa.AverageBlur(1.2)),
#         iaa.Sometimes(0.5,iaa.Sharpen(alpha=(0, 1.0), lightness=(0.65, 1.35))),
# #         iaa.Sometimes(0.5, iaa.SaltAndPepper(0.01,False)),
# #         iaa.Add((-5, 5)), # change brightness of images (by -10 to 10 of original value)
#         iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(0,0.02*255)),
#         iaa.Sometimes(0.5,iaa.GammaContrast((0.3,1.5))),
#     ])

    def __call__(self, img, mask=None):
        img = np.array(img)        
        return self.aug.augment_image(image=img)
#         return self.aug(image=img, segmentation_maps=label)
#         iaa.AddToHueAndSaturation(from_colorspace="RGB",value=(-20, 20))  #Hue-> color, saturation -> saido

trans = transforms.Compose([
        ###Color jitter : brightness=0, contrast=0, saturation=0, hue=0
        # transforms.ColorJitter(0.5,0.5,0.5,0.5),  ###strong aug  
        transforms.ColorJitter(0.1,0.0,0.1,0.1),  ###weak aug, strong aug2
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
        seg_mask = cv2.imread('./dataset/seg_train/' + train_pd['ImageId'].iloc[idx] + '.jpg',cv2.IMREAD_GRAYSCALE)
        seg_mask = np.uint8((seg_mask>0))
        img = np.array(img[:,:,::-1])
        
        if self.is_validate:
            flip = False
        else:
            flip = True if np.random.random()>flip_rate else False
        
        mask, regr = get_mask_and_regr(img, train_pd['PredictionString'][idx],flip=flip)
        img_pre = preprocess_image(img,flip)  #shape(batch,512,512), #range: [0~1]
        seg_mask = preprocess_mask_image(seg_mask,flip)
        seg_mask = np.int32((seg_mask>0))
        
        img_pil = (img_pre*255).astype('uint8')
        img_pil = Image.fromarray(img_pil)  #ndarray: Take uint8 as input, range[0~255], #imgpil -> (512,512,3), (0~255)
        img_pre_trans = self.transform(img_pil)
        regr = np.transpose(regr,(2,0,1))
        seg_mask = torch.from_numpy(seg_mask).long()
        return img_pre_trans, mask, regr, seg_mask

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


### prediction 1-7: ['pitch_cos', 'pitch_sin', 'roll', 'x', 'y', 'yaw', 'z']
from CenterNet.src.lib.models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, RegLoss_without_ind
f_loss = FocalLoss()
l1_loss = RegLoss_without_ind()

def gaussian_filter(a):
    b = torch.nn.MaxPool2d(3,stride=1,padding=1)(a)
    eq = torch.eq(a,b).float()
    a = a*eq
    return a

CEloss = torch.nn.CrossEntropyLoss()
def criterion_segmask(pred, mask, regr, seg_mask, weight=0.5, result_average=True, multiloss=True):
    #pred:(batch,8,64,64) mask:(batch,64,64) regr:(batch,7,64,64)
    scalar = 1
    pred_seg_mask = pred[:,8:10]
    zero_mask = pred[:,8].unsqueeze(1)
    one_mask = pred[:,9].unsqueeze(1)
    mul_mask = (one_mask>zero_mask).float()
    
    pred_mask = torch.sigmoid(pred[:, 0])
    # pred_mask = torch.clamp(torch.sigmoid(pred[:, 0]), min=1e-12, max=1-1e-12) #(batch,64,64)
    pred_mask = pred_mask.unsqueeze(1)
    gaussian_pred = gaussian_filter(pred_mask)
    
    ###mul_V1
    # gaussian_pred = pred_mask*mul_mask
    
    ###mul_V2
    gaussian_pred = pred_mask

    mask_loss = scalar*f_loss(gaussian_pred, mask.unsqueeze(1))
    if multiloss:
        pred_xyz = torch.cat((pred[:,4].unsqueeze(1),pred[:,5].unsqueeze(1),pred[:,7].unsqueeze(1)), 1)*mul_mask
        pred_rollyaw = torch.cat((pred[:,3].unsqueeze(1),pred[:,6].unsqueeze(1)), 1)*mul_mask
        pred_pitch = torch.cat((pred[:,1].unsqueeze(1),pred[:,2].unsqueeze(1)), 1)*mul_mask
        
        gt_xyz = torch.cat((regr[:,3].unsqueeze(1),regr[:,4].unsqueeze(1),regr[:,6].unsqueeze(1)), 1)*mul_mask
        gt_rollyaw = torch.cat((regr[:,2].unsqueeze(1),regr[:,5].unsqueeze(1)), 1)*mul_mask
        gt_pitch = torch.cat((regr[:,0].unsqueeze(1),regr[:,1].unsqueeze(1)), 1)*mul_mask
        
        xyz_loss = scalar*l1_loss(pred_xyz,gt_xyz,mask)
        rollyaw_loss = scalar*l1_loss(pred_rollyaw,gt_rollyaw,mask)
        pitch_loss = scalar*l1_loss(pred_pitch,gt_pitch,mask)
        seg_mask_loss = CEloss(pred_seg_mask,seg_mask)
        
        loss = mask_loss + xyz_loss + rollyaw_loss + pitch_loss
    else:
        regr_loss = mask_loss
        loss = mask_loss

    if not result_average:
        loss *= pred.shape[0]
        
    return loss ,mask_loss , xyz_loss, rollyaw_loss, pitch_loss, seg_mask_loss

from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.Conv2d(out_ch, out_ch, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(out_ch),
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.05),
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
        
        # input is CHW
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
        self.base_model = EfficientNet.from_pretrained('efficientnet-b5')
        
        self.conv0 = double_conv(5, 128)
        self.conv1 = double_conv(128, 256)
        self.conv2 = double_conv(256, 512)
        self.conv3 = double_conv(512, 1024)
        
        self.mp = nn.MaxPool2d(2)
        self.dp = nn.Dropout(0.5)
        
        ###eff-b5
        self.up1 = up(2050 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes+2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.dp(self.mp(self.conv0(x0)))
        x2 = self.dp(self.mp(self.conv1(x1)))
        x3 = self.dp(self.mp(self.conv2(x2)))
        x4 = self.dp(self.mp(self.conv3(x3)))
        x_center = x[:, :, :, :]
        feats = self.base_model.extract_features(x_center)    #[1, 1280, 16, 24] for eff-b0
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x

def get_kfold_dataset_loader(k=None,val_rate=None,indices_len=None, batch_size=None,num_workers=None):
    train_loader_list = []
    val_loader_list = []
    indices = np.arange(indices_len)
    val_len = indices_len//k
    idx = 0
    for i in range(k):
        ind = np.concatenate([indices[:idx],indices[idx+val_len:],indices[idx:idx+val_len]])
        idx += val_len
        train_dataset = ADDataset(data_len=len(ind),is_validate=False,validate_rate=val_rate,indices=ind)
        val_dataset = ADDataset(data_len=len(ind),is_validate=True,validate_rate=val_rate,indices=ind)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
    return train_loader_list, val_loader_list

if __name__=='__main__':
    cv_nth = 9
    batch_size = 1
    num_workers = 8
    lr = 5e-4
    lr_period = 8
    epochs = 300
    val_freq = 1
    model_ver = "./saved_model/Cross_validation_0120/fold{}_rmsprop_5e-4_b1_effb5_2048x360_SegAid_Mulv2_Leaky_weakAug_flip0.5_bgTrue_peak_noclip_multiloss".format(cv_nth)
    print(model_ver)

    train_pd = pd.read_csv("./dataset/train_remove.csv")
    indices_len = len(train_pd)
    indices = np.arange(indices_len)

    num_workers = 8
    k = 10
    # indices_len = 74340
    vr = (indices_len//k)/indices_len
    print("validation rate:",vr)
    ###K-fold dataset
    train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    train_loader = train_loaders[cv_nth]
    val_loader = val_loaders[cv_nth]

    model = MyUNet(8)
    if device=='cuda':
        model.cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=4,factor=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs, 10) * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=lr_period,T_mult=1,eta_min=8e-6) #original 

    min_loss = 100000
    multiloss = True
    loss_th = 2
    best_model_dict = None

    for ep in range(1,epochs+1):
        print("Ep",ep)
        model.train()
        train_sum = 0
        ml_sum = 0
        xyzl_sum = 0
        ryl_sum = 0
        pl_sum = 0
        sm_sum = 0
        data_num = 0
        
        for batch_idx, (img_batch, mask_batch, regr_batch, segmask_batch) in enumerate(train_loader):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            segmask_batch = segmask_batch.to(device)
            optimizer.zero_grad()
            output = model(img_batch)

            loss,mask_loss ,xyz_loss,ry_loss,p_loss,sm_loss  = criterion_segmask(output, mask_batch, regr_batch, segmask_batch, multiloss=multiloss)
            train_sum += loss.data
            ml_sum += mask_loss.data
            xyzl_sum += xyz_loss.data
            ryl_sum += ry_loss.data
            pl_sum += p_loss.data
            sm_sum += sm_loss.data
            # if batch_idx%3==0:        
            #     print('loss:{:.5f} smloss:{:.5f}  mloss:{:.5f} xyzloss{:.5f}  ryloss{:.5f} ploss{:.5f}'.
            #     format(loss.data,sm_loss.data,mask_loss.data,xyz_loss.data,ry_loss.data,p_loss.data))

            ###Way1
            loss.backward()

            optimizer.step()
            data_num += 1

        train_sum /= data_num
        ml_sum /= data_num
        xyzl_sum /= data_num
        ryl_sum /= data_num
        pl_sum /= data_num
        print('Epoch:{} lr:{:.5f} loss:{:.5f} mloss:{:.5f} xyzloss{:.5f}  ryloss{:.5f} ploss{:.5f}'.
            format(ep,optimizer.param_groups[0]['lr'],train_sum,ml_sum,xyzl_sum,ryl_sum,pl_sum))

        ###Cosine Annealing          
        # lr_scheduler.step()

        if ep%val_freq==0:
            model.eval()
            val_sum = 0
            ml_sum = 0
            xyzl_sum = 0
            ryl_sum = 0
            pl_sum = 0
            sm_sum = 0
            data_num = 0
            with torch.no_grad():
                for img_batch, mask_batch, regr_batch in val_loader:
                    img_batch = img_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    regr_batch = regr_batch.to(device)
                    output = model(img_batch)
                    loss,mask_loss ,xyz_loss,ry_loss,p_loss,sm_loss  = criterion_segmask(output, mask_batch, regr_batch, segmask_batch, multiloss=multiloss)
                    train_sum += loss.data
                    ml_sum += mask_loss.data
                    xyzl_sum += xyz_loss.data
                    ryl_sum += ry_loss.data
                    pl_sum += p_loss.data
                    sm_sum += sm_loss.data
                    data_num += 1
                    
                val_sum /= data_num
                ml_sum /= data_num
                xyzl_sum /= data_num
                ryl_sum /= data_num
                pl_sum /= data_num
                sm_sum /= data_num

            print('Val loss: {:.4f} Sm loss:{:.4f} Mask loss:{:.4f} xyz loss:{:.4f} rollyaw loss:{:.4f} pitch loss:{:.4f}'.
            format(val_sum,sm_sum,ml_sum,xyzl_sum,ryl_sum,pl_sum))
            prev_val_loss = val_sum
            
            ###ReduceLROnPlateau
            lr_scheduler.step(val_sum)
            
            if val_sum < min_loss:
                min_loss = val_sum
                best_model_dict = model.state_dict()
                path = "{}_Ep{}_loss{:.4f}".format(model_ver,ep,min_loss)
                pos = path.find("Ep")
                print(path)
                if ep>=5:
                    torch.save(best_model_dict,path)
            elif ep>=5 and ep%5 == 0:
                path = "{}_Ep{}_loss{:.4f}".format(model_ver,ep,val_sum)
                print(path)
                torch.save(model.state_dict(),path)
            
            path2 = path[:pos]+".current"
            torch.save(model.state_dict(),path2)
