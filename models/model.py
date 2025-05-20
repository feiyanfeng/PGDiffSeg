import torch.nn as nn
# import torch
# import torch.nn.functional as F
from models.model_utils import *
from models.classify_model import classify_model
# from denseblock import DenseBlock



class Unet(nn.Module):
    def __init__(self,base_channels, rate, time_emb_scale=1.0, num_groups=16, res_scale=0.2): # , map_down=False
        super().__init__()
        self.res_scale = res_scale
        #rate = [1,1,1,2,2,4,4]
        # in_ch = 1
        out_ch = 1
        ch = [base_channels] + [base_channels * i for i in rate]

        self.down_blocks_x1 = nn.ModuleList()
        self.down_blocks_x2 = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for i in range(len(ch)-1):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'down: in-{ch[i]},out-{ch[i+1]}')
            self.down_blocks_x1.append(Down_block2(ch[i], ch[i+1], base_channels, time_emb_scale,
                                    num_groups=ng))
            self.down_blocks_x2.append(Down_block2(ch[i], ch[i+1], base_channels, time_emb_scale,
                                    num_groups=ng))
            self.attention_blocks.append(PSA(ch[i+1]))
        # self.attention_blocks[-1] = nn.Sequential()

        ng = num_groups
        while ng%ch[-1] != 0 and ng != 1:
                ng//=2
        print('bottlenek, ch=', ch[-1])
        self.bottleneck = Bottleneck(ch[-1], ng)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch)-1)):
            ng = num_groups
            while (ng%ch[i+1] != 0 or ng%ch[i+1] != 0) and ng != 1:
                ng//=2
            print(f'up block: in-{ch[i+1]},out-{ch[i]}')
            self.up_blocks.append(Up_block2(ch[i+1], ch[i], base_channels, time_emb_scale,
                                  num_groups=ng))

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_ch, 3, padding=1)
        )
        
        # self.down3 = nn.Sequential()
        # if map_down:
        #     self.down3 = nn.Conv2d(ch[-1], ch[-1], 3, stride=2, padding=1)

        self.gradients = []

    
    def save_gradient(self, grad):
        # print('save grad')
       self.gradients = grad


    def cam(self, x1, x2, t, resnet_map, location):
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        
        act = None
        
        #  这是针对 4层网络
        d_x1_, d_x1 = self.down_blocks_x1[0](d_x1, t)
        d_x2_, d_x2 = self.down_blocks_x2[0](d_x2, t)
        if location == 'BA_down1':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'BA_down1_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1
        d_x1, d_x2 = self.attention_blocks[0](d_x1, d_x2)
        out_x1.append(d_x1_)
        out_x2.append(d_x2_)
        if location == 'down1':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'down1_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1
        
        d_x1_, d_x1 = self.down_blocks_x1[1](d_x1, t)
        d_x2_, d_x2 = self.down_blocks_x2[1](d_x2, t)
        if location == 'BA_down2':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'BA_down2_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1
        d_x1, d_x2 = self.attention_blocks[1](d_x1, d_x2)
        out_x1.append(d_x1_)
        out_x2.append(d_x2_)
        if location == 'down2':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'down2_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1
            
        d_x1_, d_x1 = self.down_blocks_x1[2](d_x1, t)
        d_x2_, d_x2 = self.down_blocks_x2[2](d_x2, t)
        if location == 'BA_down3':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'BA_down3_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1
        d_x1, d_x2 = self.attention_blocks[2](d_x1, d_x2)
        out_x1.append(d_x1_)
        out_x2.append(d_x2_)
        if location == 'down3':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'down3_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1
            
        d_x1_, d_x1 = self.down_blocks_x1[3](d_x1, t)
        d_x2_, d_x2 = self.down_blocks_x2[3](d_x2, t)
        if location == 'BA_down4':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'BA_down4_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1
        
        d_x1, d_x2 = self.attention_blocks[3](d_x1, d_x2)
        out_x1.append(d_x1_)
        out_x2.append(d_x2_)
        if location == 'down4':
            d_x2.register_hook(self.save_gradient)
            act = d_x2
        elif location == 'down4_2':
            d_x1.register_hook(self.save_gradient)
            act = d_x1

        m = self.bottleneck(d_x1 + d_x2)
        if location == 'bottleneck':
            m.register_hook(self.save_gradient)
            act = m
            
            
        u = m + resnet_map.mul(self.res_scale)
        
        u = self.up_blocks[0](u, out_x1[3]+out_x2[3], t)
        if location == 'up4':
            u.register_hook(self.save_gradient)
            act = u

        u = self.up_blocks[1](u, out_x1[2]+out_x2[2], t)
        if location == 'up3':
            u.register_hook(self.save_gradient)
            act = u

        u = self.up_blocks[2](u, out_x1[1]+out_x2[1], t)
        if location == 'up2':
            u.register_hook(self.save_gradient)
            act = u

        u = self.up_blocks[3](u, out_x1[0]+out_x2[0], t)
        if location == 'up1':
            u.register_hook(self.save_gradient)
            act = u
        
        o = self.out_block(u)
        return act, o

    def forward(self, x1, x2, t, resnet_map):  # x1 for xt, x2 for image
        out_x1 = []
        out_x2 = []
        d_x1_, d_x1 = None, x1
        d_x2_, d_x2 = None, x2
        for down_block_x1, down_block_x2, attention_block in zip(self.down_blocks_x1, self.down_blocks_x2, self.attention_blocks):
            d_x1_, d_x1 = down_block_x1(d_x1, t)
            d_x2_, d_x2 = down_block_x2(d_x2, t)
            d_x1, d_x2 = attention_block(d_x1, d_x2)
            out_x1.append(d_x1_)
            out_x2.append(d_x2_)

        m = self.bottleneck(d_x1 + d_x2)


        # u = m + self.down3(resnet_map).mul(self.res_scale)
        u = m + resnet_map.mul(self.res_scale)
        for up_block, d_x1_, d_x2_ in zip(self.up_blocks, reversed(out_x1), reversed(out_x2)):
            u = up_block(u, d_x1_+d_x2_, t)

        return self.out_block(u)

class denoise_model(nn.Module):
    '''
    加上SuperResnet
    '''
    def __init__(self, args, prior_args):
        super().__init__()        
        self.sdbx = SDB(1, args.base_channels, args.super_resnet_deep, res_scale=args.res_scale1)  # for image
        self.sdby = SDB(1, args.base_channels, args.super_resnet_deep, res_scale=args.res_scale1)  # for xt

        # load='/root/ffy/diffusion/my_diffusion/log/classify05/epoch-50-model.pth'
        self.prior = prior_model(prior_args)
        self.prior.to(args.device)
        if prior_args.load_prior:
            self.classify_model.load_state_dict(torch.load(prior_args.load_prior, map_location=args.device))
            print('load prior', )
        
        self.unet = Unet(args.base_channels, args.unet_rate, res_scale=args.res_scale2) # map_down=args.map_down, 

    def forward(self, x, y, t): # x-xt y-image
        classify_map = self.prior.get_map(y)
        x = self.sdbx(x)
        y = self.sdby(y)
        return self.unet(x, y, t, classify_map)



def get_denoise_model(args, classify_args):
    denoise_model = eval(args.model_name)
    return denoise_model(args, classify_args)
