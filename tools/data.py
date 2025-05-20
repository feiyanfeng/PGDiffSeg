from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np



def load_train(args):
    # get ids 
    with open(f"{args.data.root}/{args.data.train_list}.txt", "r") as f:
        train_list = f.read()
    train_list = train_list.split('\n')
    with open(f"{args.data.root}/{args.data.test_list}.txt", "r") as f:
        test_list = f.read()
    test_list = test_list.split('\n')
    
    train_set = DataGen(args.data.root, train_list)
    test_set = TestGen(args.data.root, test_list)

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, batch_size=args.training.batch_size, num_workers=args.training.num_workers)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size=args.training.batch_size*2, num_workers=args.test.num_workers)

    return train_loader, test_loader

def load_test(args):
    # get ids 
    with open(f"{args.data.root}/{args.data.test_list}.txt", "r") as f:
        test_list = f.read()
    test_list = test_list.split('\n')

    test_set = TestGen(args.data.root, test_list)

    loader_args = dict(batch_size=args.test.batch_size, num_workers=args.test.num_workers)
    return DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

class DataGen(Dataset):
    def __init__(self, root, name_list):
        self.root = root
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}/{self.name_list[index]}', allow_pickle=True)
        return data[0], data[1]  # npy = [image, mask]
    
    def __len__(self):
        return len(self.name_list)
    

class TestGen(Dataset):
    def __init__(self, root, name_list):
        self.root = root
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}/{self.name_list[index]}', allow_pickle=True)
        return self.name_list[index][:-4], data[0] , data[1] # npy = [image, mask]
    
    def __len__(self):
        return len(self.name_list)