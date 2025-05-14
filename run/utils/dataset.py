import json
import os
from os import path as osp
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms

class VITONDataset(data.Dataset):
    def __init__(self, args,type):
        super(VITONDataset, self).__init__()
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.semantic_nc = args.semantic_nc
        self.data_path = osp.join(args.dataset_dir,type)
        self.transform = transforms.Compose([transforms.ToTensor(),
        ])
        self.order = args.test_order
        self.img_names = [] # model image
        self.garm_names = []  # cloth image

        with open(osp.join(args.dataset_dir,f"{type}_pairs.txt"), 'r') as f:
            for line in f.readlines():
                if type == 'train': # TODO 训练模式
                    img_name, garm_name = line.strip().split()
                    garm_name = img_name
                else: # TODO 测试模式
                    if self.order == 'paired':
                        img_name, garm_name = line.strip().split()
                        garm_name = img_name
                    else:
                        img_name, garm_name = line.strip().split()
                        
                self.img_names.append(img_name)
                self.garm_names.append(garm_name)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        garm_name = self.garm_names[index]
        garm = Image.open(osp.join(self.data_path, 'cloth', garm_name)).convert('RGB')
        garm = transforms.Resize(self.img_width, interpolation=2)(garm)
        garm_mask = Image.open(osp.join(self.data_path, 'cloth-mask', garm_name))
        garm_mask = transforms.Resize(self.img_width, interpolation=0)(garm_mask)

        garm = self.transform(garm)  # [0,1] 没有normalize
        garm_mask_array = np.array(garm_mask)
        garm_mask_array = (garm_mask_array >= 128).astype(np.float32)
        garm_mask = torch.from_numpy(garm_mask_array)  # [0,1]
        garm_mask.unsqueeze_(0)

        # load person image 原图
        img_pil_big = Image.open(osp.join(self.data_path, 'image', img_name))
        img_pil = transforms.Resize(self.img_width, interpolation=2)(img_pil_big) # 对原图resize为512
        img = self.transform(img_pil)      
        
        # TODO 直接加载 masked vton image, 减少mask时间
        img_agnostic = Image.open(osp.join(self.data_path, 'agnostic-v3.3', img_name))
        img_agnostic = transforms.Resize(self.img_width, interpolation=2)(img_agnostic)
        img_agnostic = self.transform(img_agnostic) # TODO check一下有没有归一化, ToTensor应该是映射到[0,1]
        # print("===========================================img_agnostic.shape2222 = ", img_agnostic.shape)

        # TODO 直接加载模特衣服的mask, 用于后续训练
        agnostic_mask = Image.open(osp.join(self.data_path, 'image_mask', img_name))
        agnostic_mask = transforms.Resize(self.img_width, interpolation=2)(agnostic_mask)
        agnostic_mask = self.transform(agnostic_mask) # TODO check一下有没有归一化，ToTensor应该是映射到[0,1]
        # print("===========================================agnostic_mask.shape333 = ", agnostic_mask.shape)
        
        
        # load caption
        caption_name = garm_name.replace(".jpg", ".txt")
        caption_path = osp.join(self.data_path, 'cloth_caption', caption_name)
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as file:
                caption = file.read()
        else:
            # print("File does not exist. ", caption_name) # TODO 这个描述是干什么用的，需要吗
            caption = "A cloth"
            # caption = "" # TODO edit设置为空试试

        result = {
            'img_name': img_name,
            'garm_name': garm_name,
            'img_ori': img,
            'img_vton': img_agnostic,
            'img_vton_mask': agnostic_mask, # TODO 新增模特的mask
            'img_garm': garm,
            'img_garm_mask': garm_mask, # TODO 新增衣服的mask
            "prompt":caption
        }
        return result

    def __len__(self):
        return len(self.img_names)

class VITONDataLoader:
    def __init__(self, args, dataset):
        super(VITONDataLoader, self).__init__()
        train_sampler = data.sampler.RandomSampler(dataset)
        self.data_loader = data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
        self.batch_size = args.batch_size
        
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch
    
class DressCodeDataset(data.Dataset):
    def __init__(self, args,type):
        super(DressCodeDataset, self).__init__()
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.semantic_nc = args.semantic_nc
        self.data_path = osp.join(args.dataset_dir)
        self.transform = transforms.Compose([transforms.ToTensor(),
        ])

        # load data list
        self.data_list=[]
        with open(osp.join(args.dataset_dir,f"{type}_pairs_paired.txt"), 'r') as f:
            for line in f.readlines():
                img_name, garm_name,label = line.strip().split()
                self.data_list.append((img_name,garm_name,label))
        self.img_names = [x[0] for x in self.data_list]
        self.garm_names = [x[1] for x in self.data_list]
        self.labels = [x[2] for x in self.data_list] 
        self.label2idx = {0:'upper_body',1:'lower_body',2:'dresses'}
        self.label2category = {0:'upperbody',1:'lowerbody',2:'fullbody'}
        self.label_map={
            "background": 0,
            "hat": 1,
            "hair": 2,
            "sunglasses": 3,
            "upper_clothes": 4,
            "skirt": 5,
            "pants": 6,
            "dress": 7,
            "belt": 8,
            "left_shoe": 9,
            "right_shoe": 10,
            "head": 11,
            "left_leg": 12,
            "right_leg": 13,
            "left_arm": 14,
            "right_arm": 15,
            "bag": 16,
            "scarf": 17,
        }
        self.keypoints_map={
            0.0: "nose",
            1.0: "neck",
            2.0: "right shoulder",
            3.0: "right elbow",
            4.0: "right wrist",
            5.0: "left shoulder",
            6.0: "left elbow",
            7.0: "left wrist",
            8.0: "right hip",
            9.0: "right knee",
            10.0: "right ankle",
            11.0: "left hip",
            12.0: "left knee",
            13.0: "left ankle",
            14.0: "right eye",
            15.0: "left eye",
            16.0: "right ear",
            17.0: "left ear"
        }
            
    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.img_width, self.img_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic
            
    def __getitem__(self, index):
        img_name = self.img_names[index]
        garm_name = self.garm_names[index]
        label = int(self.labels[index])
        garm = Image.open(osp.join(self.data_path, self.label2idx[label], "images",garm_name)).convert('RGB')
        garm = transforms.Resize(self.img_width, interpolation=2)(garm)
        garm = self.transform(garm) 
    
        # load pose image
        pose_name = img_name.replace('0.jpg', '5.jpg')
        pose_rgb = Image.open(osp.join(self.data_path,self.label2idx[label] ,'skeletons', pose_name))
        pose_rgb = transforms.Resize(self.img_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        # load pose data
        pose_name = img_name.replace('0.jpg', '2.json')
        with open(osp.join(self.data_path,self.label2idx[label] ,'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data[:, :2]

        # load parsing image
        parse_name = img_name.replace('0.jpg', '4.png')
        parse_big = Image.open(osp.join(self.data_path,self.label2idx[label] ,'label_maps', parse_name))
        
        # load person image  
        img_pil_big = Image.open(osp.join(self.data_path, self.label2idx[label],'images', img_name))
        img_pil = transforms.Resize(self.img_width, interpolation=2)(img_pil_big)
        img = self.transform(img_pil)      
        
        # get masked vton image
        mask, mask_gray = get_mask_location("dc", self.label2idx[label], parse_big, pose_data)    
        masked_vton_img = Image.composite(mask_gray,img_pil,mask)
        masked_vton_img = self.transform(masked_vton_img)
        
        result = {
            'img_name': img_name,
            'garm_name': garm_name,
            'img_ori': img,
            'img_vton': masked_vton_img,
            'pose': pose_rgb,
            'img_garm': garm,
            'prompt': self.label2category[label],
            "pixel_values": img
        }
        return result

    def __len__(self):
        return len(self.img_names)

class DressCodeDataLoader:
    def __init__(self, args, dataset):
        super(DressCodeDataLoader, self).__init__()
        train_sampler = data.sampler.RandomSampler(dataset)
        self.data_loader = data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
        self.batch_size = args.batch_size
        
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
