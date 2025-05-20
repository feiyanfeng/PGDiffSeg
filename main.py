import sys
import traceback
import datetime

from tools.configs import setting
from tools.logger import Logger
from trainer import Trainer

import os
import torch


args = setting()
# print(args)
# exit()
log_name = args.log.name if args.log.name else args.name
logger = Logger(f'log/{log_name}.log')

def print_namespace():
    for arg_name, arg_value in vars(args).items(): logger.info(f'### {arg_name}: {arg_value}', '%(message)s')


def main():
    
    logger.info(f'{args.mode} to {log_name}\n')
    print_namespace()

    logger.info(f'\n{args.mode} start...\n')
    # logger.info(args, '%(message)s\n')
    # print(args.data)

    try:
        runner = Trainer(args, logger)
        if args.mode=='train':
            runner.train()
        elif args.mode=='test':
            for i in range(args.test.ensemble_num):
                runner.test()
        else:
            raise NotImplementedError(f'{args.mode} not matched')
        logger.info('ending...\n\n\n')
         
    except Exception:
        # print(traceback.format_exc())
        logger.error(traceback.format_exc())
        if args.mode=='train':
            states = [
                    runner.model.state_dict(),
                    runner.optimizer.state_dict(),
                    runner.epoch
                ]
            if args.model.ema:
                states.append(runner.ema_helper.state_dict())
            torch.save(states, os.path.join(args.training.weigth_path, f"{args.name}_exception.pth"))
    
    return 0



if __name__ == '__main__':
    sys.exit(main())

