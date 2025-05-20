import os
import numpy as np
# import numpy as np
import torch
# from torch import nn
from tools.ema import EMA
from models.diffusion import diffusion_model
from sample_solver.dpmsolver import DPM_Solver_evaluate
# from sample_solver.heun import heun_evaluate
from sample_solver.uni_pc import uni_pc_evaluate
# from sample_solver.DPM_Solver_v3 import DPM_Solver_v3_evaluate

from tools.data import load_train, load_test
from tools.evaluation import  Evaluation
from tools.utils import Optimizer
import SimpleITK as sitk
from datetime import datetime  


class Trainer:
    def __init__(self, args, logger):
        self.logger = logger
        self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        args.diffusion.device = self.device
        args.model.device = self.device
        
        # model
        if not (args.mode=='test' and args.test.pred_dir):
            self.model = diffusion_model(args).to(self.device)
        # self.num_timesteps = args.diffusion.timesteps

        self.best_dice = 0.0
        self.best_epoch = 0
        
        #  save weight
        self.weight_path = args.training.weigth_path+'/'+args.name
        os.makedirs(self.weight_path, exist_ok=True)

        self.args = args


    @torch.no_grad()
    def evaluate(self, loader):
        model = self.model
        model.eval()
        e = Evaluation(None, self.logger, self.args.evaluation.choice, Global='only', resize=self.args.evaluation.resize)
        # if save: os.makedirs(save, exist_ok=True)
        for names, image_ , label in loader:
            image_ = image_.to(self.device)
            #print('----------------------', image.shape)
            samples = model.sample(image_)
            # save
            samples = samples.numpy().clip(-1, 1).squeeze()
            label = label.numpy().squeeze()

            for mask, name, label_ in zip(samples, names, label):
                # print(mask.shape, label_.shape)
                e.cacu(name, mask, label_.squeeze())
                # if save: sitk.WriteImage(sitk.GetImageFromArray(mask), save + '/' + name+'.nii')
        # if save: e.save(f'result.csv')
        return e.view(get_dice='Global')
    
    @torch.no_grad()
    def evaluate_save(self, loader, save = None):
        now = datetime.now().strftime("%y.%m.%d-%H%M%S")
        if save is None:
            save = f'sample/{self.args.name}_{now}'
            # self.logger.info(f'You seem to have forgotten to pass in the save path, see {save}', '%(message)s')
            os.makedirs(save)
        self.logger.info(f'predict results saved to path:{save}', '%(message)s')
        model = self.model
        model.eval()
        e = Evaluation(save, self.logger, self.args.evaluation.choice, Global=True, resize=self.args.evaluation.resize)
        l = len(loader)
        for i, (names, image_ , label) in enumerate(loader):
            image_ = image_.to(self.device)
            #print('----------------------', image.shape)
            samples = model.sample(image_)
            # save
            samples = samples.numpy().clip(-1, 1).squeeze()
            label = label.numpy().squeeze()
            self.logger.info(f'{i}/{l}', '%%%%%%---%(message)s---%%%%%%  %(asctime)s')  # %%%---1/4---%%%  

            for mask, name, label_ in zip(samples, names, label):
                # print(mask.shape, label_.shape)
                e.cacu(name, mask, label_.squeeze())
                sitk.WriteImage(sitk.GetImageFromArray(mask), save + '/' + name+'.nii')
            
        e.save(f'result_{now}.csv')


    def train(self):
        args = self.args
        # self.diffusion.num_timesteps = self.args.diffusion.timesteps
        # data
        train_loader, val_loader = load_train(args)
        # self.val_loader = val_loader
        # optim
        optim = Optimizer()
        self.optimizer = optim.get_optimizer(args.optim, self.model.parameters())

        # criterion = F.mse_loss
        if args.model.ema:
            self.ema_helper = EMA(mu=args.model.ema_rate)
            self.ema_helper.register(self.model)
        else:
            self.ema_helper = None
        # load
        start_epoch = 1
        if args.load:
            states = torch.load(args.load, map_location=self.device)
            self.model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = args.optim.eps
            self.optimizer.load_state_dict(states[1])
            start_epoch = states[2]+1
            if args.model.ema:
                self.ema_helper.load_state_dict(states[3])
        
        # train
        self.logger.info('training...')
        for epoch in range(start_epoch, args.training.epochs+1):
            self.model.train()
            total_loss = 0
            for i, (image, mask) in enumerate(train_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                if args.optim.lr_schedule:
                    optim.adjust_learning_rate(self.optimizer, i / len(train_loader) + epoch, args)
                loss = self.model(mask, image)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if args.model.ema:
                    self.ema_helper.update(self.model)
                # exit()
            self.epoch = epoch
            ave_loss = total_loss/len(train_loader)
            self.logger.info(f'epoch {epoch}, ave_loss={ave_loss}')
            self.epoch_process(epoch, val_loader)
        
        self.train_end(epoch, val_loader)
        

    def epoch_process(self, epoch, loader):
        if (epoch > self.args.training.sample_checkpoint and epoch % self.args.training.sample_rate == 0) or (epoch in self.args.training.sample_some):
            dice = self.evaluate(loader)
            if dice > self.best_dice:
                self.logger.info(f'save best: {dice}')
                self.best_dice = dice
                self.best_epoch = epoch
                states = [
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                        epoch
                    ]
                if self.args.model.ema:
                    states.append(self.ema_helper.state_dict())
                torch.save(states, os.path.join(self.weight_path, "best_ckpt.pth"))
        return


    def train_end(self, epoch, loader):
        # Training completed
        if self.args.evaluation.acc_sample:
            self.logger.info(f'training over, best_dice {self.best_dice} in epoch {self.best_epoch}')
            print(f'best_dice: {self.best_dices}, epoch:{self.epoch_index}, {datetime.now()}')
        else:
            self.logger.info(f'training over, best_dice {self.best_dice} in epoch {self.best_epoch}')
            print(f'best_dice: {self.best_dice}, epoch:{self.best_epoch}, {datetime.now()}')

        # save last
        states = [
                self.model.state_dict(),
                self.optimizer.state_dict(),
                epoch
            ]
        if self.args.model.ema:
            states.append(self.ema_helper.state_dict())
        torch.save(states, os.path.join(self.weight_path, f"epoch{epoch}_ckpt.pth"))
        
        if epoch < self.args.training.sample_checkpoint or epoch % self.args.training.sample_rate != 0: self.evaluate_save(loader)
        return


    def read_list(self, txt_name):
        with open(txt_name, "r") as f:
            train_list = f.read()
        return train_list.split('\n')
    
    def load(self): # load model, only for test
        states = torch.load(self.args.load, map_location=self.device)
        self.model.load_state_dict(states[0])
        self.logger.info(f'---in epoch {states[2]}---', '%(message)s')

    def test(self, load=False):
        if self.args.test.pred_dir: # 已经有预测结果了
            root = self.args.data.root
            pred_dir = self.args.test.pred_dir
            name_list = self.read_list(f"{root}/{self.args.data.test_list}.txt")
            e = Evaluation(pred_dir, self.logger, self.args.evaluation.choice, Global=True, resize=self.args.evaluation.resize)
            for name in name_list:
                try:
                    data = np.load(f'{root}/{name}', allow_pickle=True)
                    gt = data[1][0]
                    pred = sitk.GetArrayFromImage(sitk.ReadImage(f'{pred_dir}/{name[:-4]}.nii'))
                    e.cacu(name[:-4], pred, gt)
                except Exception as error:
                    print(error)
                    self.logger.error(error)
                    break
            e.save(f'result_{datetime.now().strftime("%y.%m.%d-%H%M%S")}.csv')
        else: # load
            if not load: self.load()
            test_loader = load_test(self.args)
            if self.args.evaluation.acc_sample:
                self.evaluate_acc(test_loader)
            else:
                self.evaluate_save(test_loader)
        self.logger.info('-'*10, '%(message)s')

