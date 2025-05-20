import argparse


def process_args(args):
    '''
    生成新的命名空间，通过.将命名空间划分为多个子命名空间
    如 args.data.name变成args.data.name
    使用限制：
        只能划分二级子空间, 即A.B
        A.B.C不适用
    '''  
    new_args = argparse.Namespace()

    for arg_name, arg_value in vars(args).items():
        if '.' in arg_name:
            nested_args = arg_name.split('.')
            current_namespace = new_args

            for namespace_name in nested_args[:-1]:
                if not hasattr(current_namespace, namespace_name):
                    setattr(current_namespace, namespace_name, argparse.Namespace())
                current_namespace = getattr(current_namespace, namespace_name)

            setattr(current_namespace, nested_args[-1], arg_value)
        else:
            setattr(new_args, arg_name, arg_value)

    return new_args




def get_configs():
    parser = argparse.ArgumentParser(description='initial')
    # description
    parser.add_argument('--name', type=str, required=True, help='tag of this running, like busi-train01')
    # mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample_solver'])  # sample_solver：几种采样方式进行对比


    # data
    # parser.add_argument('--data.name', type=str, default='busi', help='data name') # 暂时没用着
    parser.add_argument('--data.root', type=str, required=True, help='dataset path')
    parser.add_argument('--data.train_list', type=str, required=True)
    parser.add_argument('--data.test_list', type=str, required=True)
    parser.add_argument('--data.img_size', default=128, type=int)

    # parser.add_argument('--data.num_classes', type=int, default=3)
    # Dimension : 2 (b*c*h*w)

    # model
    # parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--model.model_name', required=True, type=str, help='denoise_model1')
    parser.add_argument('--model.input_channels', type=int, default=1)
    parser.add_argument('--model.ema', type=bool, default=True)
    parser.add_argument('--model.ema_rate', type=float, default=0.9999)
    parser.add_argument('--model.super_resnet_deep', default=3, type=int)
    # parser.add_argument('--model.growth_rate', default=16, type=int)
    parser.add_argument('--model.unet_rate', required=True, type=int, nargs="+")
    parser.add_argument('--model.base_channels', required=True, type=int)
    parser.add_argument('--model.res_scale1', default=0.2, type=float)
    parser.add_argument('--model.res_scale2', default=0.2, type=float)
    # parser.add_argument('--model.load_classify', default=True, type=bool)
    # diffusion
    parser.add_argument('--diffusion.beta_schedule', type=str, default='linear')
    parser.add_argument('--diffusion.beta_start', type=float, default=0.0001)
    parser.add_argument('--diffusion.beta_end', type=float, default=0.02)
    parser.add_argument('--diffusion.timesteps', type=int, default=100)
    
    # parser.add_argument('--diffusion.test_timesteps', type=int, default=250)
    # classify
    parser.add_argument('--classify.load_resnet', type=str, default='', help='Load model from a .pth file')
    # parser.add_argument('--classify.use_classify', action='store_true')
    parser.add_argument('--classify.classify_model', help='model name')
    parser.add_argument('--classify.res_base_channels', type=int)
    parser.add_argument('--classify.resnet_rate', type=str)
    parser.add_argument('--classify.res_block', type=str)  # default: BasicBlock
    parser.add_argument('--classify.num_blocks', type=str)  # default: [2]*len(rate)
    parser.add_argument('--classify.classes', default=2, type=int, help='class number')

    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file, for training or testing')
    # training
    parser.add_argument('--training.batch_size', type=int, default=32)
    parser.add_argument('--training.epochs', type=int, default=500)
    parser.add_argument('--training.num_workers', type=int, default=1)
    parser.add_argument('--training.warmup_epochs', type=int, default=40) # Decay the learning rate with half-cycle cosine after warmup
    parser.add_argument('--training.sample_checkpoint', type=int, default=300)
    parser.add_argument('--training.sample_rate', type=int, default=5)
    parser.add_argument('--training.sample_some', default=[100, 200], type=int, nargs="+", help='In addition to the above choices, which epochs do you want to sample for')
    parser.add_argument('--training.weigth_path', type=str, default='weigths')
    # optim
    parser.add_argument('--optim.optimizer', type=str, default='Adam')
    parser.add_argument('--optim.lr', type=float, default=1e-4)
    parser.add_argument('--optim.weight_decay', type=float, default=0.00)
    parser.add_argument('--optim.beta1', type=float, default=0.9)
    parser.add_argument('--optim.amsgrad', type=bool, default=False)
    parser.add_argument('--optim.eps', type=float, default=1e-8)
    parser.add_argument('--optim.lr_schedule', type=bool, default=False)
    parser.add_argument('--optim.min_lr', type=float, default=0.0) # used when lr_schedule
    #  grad_clip: 1.0, 梯度裁剪？


    # test
    parser.add_argument('--test.ensemble_num', type=int, default=1, help='重复测试多少次, 默认只测试一次。ensemble_num>1时, 不要自定义预测结果保存位置, 否则结果会覆盖')
    parser.add_argument('--test.batch_size', type=int, default=64)
    parser.add_argument('--test.num_workers', type=int, default=0)
    parser.add_argument('--test.pred_dir', type=str)


    # evaluation
    parser.add_argument('--evaluation.choice', type=float, default=0)
    parser.add_argument('--evaluation.resize', type=int, default=0, help='对预测结果resize, 一般是用不着的, 主要是用来计算Swin-Unet的结果')
    parser.add_argument('--evaluation.save_num', type=int, default=3, help='保存前几个最好的结果, 仅acc_sample=True时有效')


    # log
    parser.add_argument('--log.name', type=str, default=None, help='name of log file, None means args.name')


    return parser.parse_args()
    



def setting():
    args = process_args(get_configs())

    args.classify.img_size = args.data.img_size
    # if args.model.load_classify: assert args.classify.load_resnet, 'classify model should preload'
    if args.classify.res_base_channels is None:
        args.classify.res_base_channels = args.model.base_channels
    if args.classify.resnet_rate is None:
        args.classify.resnet_rate = args.model.unet_rate
    # print(args)
    # print('ema:', args.model.ema)
    # exit()
    return args

if __name__ == '__main__':
    print(setting())