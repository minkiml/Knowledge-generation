import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging
from merges.dataset_merge.AutoAugmentation.autoaugment import CIFAR10Policy, SVHNPolicy
from merges.dataset_merge.label_restructring import help_label_restructuring
# Discrete sequence exp datasets: sMNIST, pMNIST, sCIFAR-10,

class FlattenAndPerm(object):
    def __init__(self, perm_f, perm_ = False):
        self.perm_f = perm_f
        self.perm_ = perm_
    def __call__(self, x):
        c = x.shape[0]
        x = x.view(c, -1).transpose(1,0)
        if self.perm_:
            x = x[self.perm_f]
        return x

class Sequential_dataset(object):
    def __init__(
                # Loaded dataset
                self, 
                path = "",
                sub_dataset = "MNIST",

                permute_ = False,
                sqs = True,
                batch_training = 32,
                batch_testing = 32,
                num_workers = 0,
                
                transform_set = {"random_crop": False,
                                 "autoaug": False,
                                 "random_crop_pase": False,
                                 "resize": False},
                
                multitasking = False

                                ):
        # Global logger
        log_loc = os.environ.get("log_loc")
        root_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) # go to one root above
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('From Sequential_dataset')

    
        self.path = path
        self.sub_dataset = sub_dataset
        self.num_workers = num_workers
        self.batch_training = batch_training
        self.batch_testing = batch_testing
        self.permute_ = permute_
        self.sqs = sqs
        self.transform_set = transform_set
        self.multitasking = multitasking

        self.imageresize = 64
        self.__read_and_construct__() 
    def __read_and_construct__(self):
        # Check if the dataset already exists in the root directory
        if os.path.exists(os.path.join(self.path,self.sub_dataset, 'raw', 'train-images-idx3-ubyte.gz')) or \
            os.path.exists(os.path.join(self.path, 'cifar-10-python.tar.gz')) or \
            os.path.exists(os.path.join(self.path, 'cifar-100-python.tar.gz')) or \
            os.path.exists(os.path.join(self.path, 'train_32x32.mat')) or \
            os.path.exists(os.path.join(self.path, 'test_32x32.mat')): ## add here
            download_ = False
        else:
            download_ = True
            self.logger.info(f"Require downloading dataset")
            
        if self.sub_dataset == "MNIST":
            self.L_ = int(28*28)
            self.C_ = 1
            self.num_class = 10
            perm_f = torch.randperm(self.L_)
            # transforms.Lambda(lambda x: x.view(-1, 1))
            if self.sqs:
                train_sets = torchvision.datasets.MNIST(root=self.path, train=True, 
                                                        transform=transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.1307,), (0.3081,)),
                                                                                    FlattenAndPerm(perm_f, self.permute_)]), 
                                                        download=download_)

                test_sets = torchvision.datasets.MNIST(root=self.path, train=False, 
                                            transform=transforms.Compose([transforms.ToTensor(), 
                                                                        transforms.Normalize((0.1307,), (0.3081,)),
                                                                        FlattenAndPerm(perm_f, self.permute_)]), 
                                            download=download_)
            else:
                train_sets = torchvision.datasets.MNIST(root=self.path, train=True, 
                                                        transform=transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.1307,), (0.3081,))]), 
                                                        download=download_)

                test_sets = torchvision.datasets.MNIST(root=self.path, train=False, 
                                            transform=transforms.Compose([transforms.ToTensor(), 
                                                                        transforms.Normalize((0.1307,), (0.3081,))]), 
                                            download=download_)
        elif self.sub_dataset == "CIFAR-10":
            self.L_ = int(32*32) if not self.transform_set["resize"] else int(self.imageresize * self.imageresize)
            self.C_ = 3
            self.num_class = 10 # TODO: check
            perm_f = torch.randperm(self.L_)
            mean_ = [0.4914, 0.4822, 0.4465] #[0.4914, 0.4822, 0.4465]  [0.5071, 0.4867, 0.4408]
            std_ = [0.2023, 0.1994, 0.2010]  # [0.2023, 0.1994, 0.2010]  [0.2675, 0.2565, 0.2761]    [0.2470, 0.2435, 0.2616] 
            # Transform
            train_transform = []
            test_transform = []
            if self.transform_set["resize"]:
                train_transform += [
                                transforms.Resize((self.imageresize, self.imageresize)) ]
                test_transform += [
                                transforms.Resize((self.imageresize, self.imageresize)) ]
                self.logger.info(f"data resizing to ({self.imageresize}, {self.imageresize}) is applied for both training and testing data")
            if self.transform_set["random_crop_pase"]:
                train_transform += [
                                transforms.RandomHorizontalFlip()]
                self.logger.info("data augmentation: RandomHorizontalFlip is applied")
            
            if self.transform_set["random_crop"]:
                train_transform += [
                                transforms.RandomCrop(size=32, padding=4) ]
                self.logger.info("data augmentation: RandomCrop is applied")
            if self.transform_set["autoaug"]:
                train_transform.append(CIFAR10Policy())
                self.logger.info("data augmentation: autoaugmentation is applied")
                # train_transform += [
                #                 transforms.RandomHorizontalFlip() ]
            
            train_transform += [
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_, std=std_)
                                ]
            test_transform += [
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_, std=std_)
                                ]
            if self.sqs:
                train_transform += [FlattenAndPerm(perm_f, self.permute_)]
                test_transform += [FlattenAndPerm(perm_f, self.permute_)]
            train_transform = transforms.Compose(train_transform)
            test_transform = transforms.Compose(test_transform)

            # if self.sqs:
            #     train_sets = torchvision.datasets.CIFAR10(root=self.path, train=True, 
            #                                 transform= train_transform, 
            #                                 download=download_)

            #     test_sets = torchvision.datasets.CIFAR10(root=self.path, train=False, 
            #                     transform= test_transform, 
            #                     download=download_)
            # else:
            train_sets = torchvision.datasets.CIFAR10(root=self.path, train=True, 
                                        transform = train_transform,  # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                                        download=download_)

            test_sets = torchvision.datasets.CIFAR10(root=self.path, train=False, 
                            transform= test_transform, 
                            download=download_)

        elif self.sub_dataset == "CIFAR-100":
            self.L_ = int(32*32)
            self.C_ = 3
            self.num_class = 100 # TODO: check
            perm_f = torch.randperm(self.L_)
            mean_ = [0.4914, 0.4822, 0.4465] #[0.4914, 0.4822, 0.4465]  [0.5071, 0.4867, 0.4408] [0.485, 0.456, 0.406]
            std_ = [0.2470, 0.2435, 0.2616] # [0.2023, 0.1994, 0.2010]  [0.2675, 0.2565, 0.2761] [0.229, 0.224, 0.225]
            # Transform
            train_transform = []
            test_transform = []
            # Transform
            if self.transform_set["resize"]:
                train_transform += [
                                transforms.Resize((self.imageresize, self.imageresize)) ]
                test_transform += [
                                transforms.Resize((self.imageresize, self.imageresize)) ]
                self.logger.info(f"data resizing to ({self.imageresize}, {self.imageresize}) is applied for both training and testing data")

            if self.transform_set["random_crop"]:
                train_transform += [
                                transforms.RandomCrop(size=32, padding=4) ]
                self.logger.info("data augmentation: random_crop is applied")
            if self.transform_set["autoaug"]:
                train_transform.append(CIFAR10Policy())
                self.logger.info("data augmentation: autoaugmentation is applied")
                # train_transform += [
                #                 transforms.RandomHorizontalFlip() ]
                
            if self.transform_set["random_crop_pase"]:
                pass
            
            train_transform += [
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_, std=std_)
                                ]
            test_transform += [
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_, std=std_)
                                ]
            if self.sqs:
                train_transform += [FlattenAndPerm(perm_f, self.permute_)]
                test_transform += [FlattenAndPerm(perm_f, self.permute_)]
            train_transform = transforms.Compose(train_transform)
            test_transform = transforms.Compose(test_transform)

            # if self.sqs:
            #     train_sets = torchvision.datasets.CIFAR10(root=self.path, train=True, 
            #                                 transform= train_transform, 
            #                                 download=download_)

            #     test_sets = torchvision.datasets.CIFAR10(root=self.path, train=False, 
            #                     transform= test_transform, 
            #                     download=download_)
            # else:
            train_sets = torchvision.datasets.CIFAR100(root=self.path, train=True, 
                                        transform = train_transform,  # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                                        download=download_)

            test_sets = torchvision.datasets.CIFAR100(root=self.path, train=False, 
                            transform= test_transform, 
                            download=download_)
        elif self.sub_dataset == "SVHN":
            self.L_ = int(32*32) if not self.transform_set["resize"] else int(self.imageresize * self.imageresize)
            self.C_ = 3
            self.num_class = 10 # TODO: check
            perm_f = torch.randperm(self.L_)
            mean_ = [0.485, 0.456, 0.406] #[0.4914, 0.4822, 0.4465]  [0.5071, 0.4867, 0.4408] [0.485, 0.456, 0.406]
            std_ = [0.229, 0.224, 0.225] # [0.2023, 0.1994, 0.2010]  [0.2675, 0.2565, 0.2761] [0.229, 0.224, 0.225]
            # Transform
            train_transform = []
            test_transform = []
            if self.transform_set["resize"]:
                train_transform += [
                                transforms.Resize((self.imageresize, self.imageresize)) ]
                test_transform += [
                                transforms.Resize((self.imageresize, self.imageresize)) ]
                self.logger.info(f"data resizing to ({self.imageresize}, {self.imageresize}) is applied for both training and testing data")

            if self.transform_set["random_crop"]:
                train_transform += [
                                transforms.RandomCrop(size=32, padding=4) ]
                self.logger.info("data augmentation: random_crop is applied")
            if self.transform_set["autoaug"]:
                train_transform.append(SVHNPolicy())
                self.logger.info("data augmentation: autoaugmentation is applied")
                # train_transform += [
                #                 transforms.RandomHorizontalFlip() ]
                
            if self.transform_set["random_crop_pase"]:
                pass
            
            
            train_transform += [
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_, std=std_)
                                ]
            test_transform += [
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_, std=std_)
                                ]
            if self.sqs:
                train_transform += [FlattenAndPerm(perm_f, self.permute_)]
                test_transform += [FlattenAndPerm(perm_f, self.permute_)]
            train_transform = transforms.Compose(train_transform)
            test_transform = transforms.Compose(test_transform)

            # if self.sqs:
            #     train_sets = torchvision.datasets.CIFAR10(root=self.path, train=True, 
            #                                 transform= train_transform, 
            #                                 download=download_)

            #     test_sets = torchvision.datasets.CIFAR10(root=self.path, train=False, 
            #                     transform= test_transform, 
            #                     download=download_)
            # else:
            train_sets = torchvision.datasets.SVHN(root=self.path, split= 'train', 
                                        transform = train_transform,  # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                                        download=download_)

            test_sets = torchvision.datasets.SVHN(root=self.path, split='test', 
                            transform= test_transform, 
                            download=download_)
            
        if self.multitasking:
            set_a_num = self.num_class // 2
            set_b_num = self.num_class - set_a_num
            [train_sets_A, test_sets_A, label_subset_A], [train_sets_B, test_sets_B, label_subset_B] = help_label_restructuring(train_sets, test_sets,
                                                                                                total_label_num= self.num_class,
                                                                                                split_p= [set_a_num, set_b_num])

            # Create a DataLoader to efficiently load the dataset in batches
            self.training_loader_A = DataLoader(train_sets_A, 
                                            batch_size=self.batch_training, 
                                            shuffle=True,
                                            num_workers= self.num_workers,
                                            drop_last=True)

            self.testing_loader_A = DataLoader(test_sets_A, 
                                            batch_size=self.batch_testing, 
                                            shuffle=False,
                                            num_workers= self.num_workers,
                                            drop_last=False)
            
            # Create a DataLoader to efficiently load the dataset in batches
            self.training_loader_B = DataLoader(train_sets_B, 
                                            batch_size=self.batch_training, 
                                            shuffle=True,
                                            num_workers= self.num_workers,
                                            drop_last=True)

            self.testing_loader_B = DataLoader(test_sets_B, 
                                            batch_size=self.batch_testing, 
                                            shuffle=False,
                                            num_workers= self.num_workers,
                                            drop_last=False)
            self.num_class = set_a_num

        else:
            # Create a DataLoader to efficiently load the dataset in batches
            self.training_loader_A = DataLoader(train_sets, 
                                            batch_size=self.batch_training, 
                                            shuffle=True,
                                            num_workers= self.num_workers,
                                            drop_last=True)

            self.testing_loader_A = DataLoader(test_sets, 
                                            batch_size=self.batch_testing, 
                                            shuffle=True,
                                            num_workers= self.num_workers,
                                            drop_last=False)
            
            self.training_loader_B = None
            self.testing_loader_B = None

            del train_sets, test_sets

        self.logger.info(f"Training {self.sub_dataset} data loader is constructed. The total number of mini-batches: {len(self.training_loader_A)}")
        self.logger.info(f"Testing {self.sub_dataset} data loader is constructed. The total number of mini-batches: {len(self.testing_loader_A)}")

        if self.multitasking:
            self.logger.info(f"Training 'subset B' {self.sub_dataset} data loader is constructed. The total number of mini-batches: {len(self.training_loader_B)}")
            self.logger.info(f"Testing 'subset B' {self.sub_dataset} data loader is constructed. The total number of mini-batches: {len(self.testing_loader_B)}")
            self.logger.info(f"The subset A has labels {label_subset_A} and the subset B has labels {label_subset_B}")

        self.logger.info(f"No validation dataset")

    def __get_training_loader__(self):
        return self.training_loader_A, self.training_loader_B
    def __get_testing_loader__(self):
        return self.testing_loader_A, self.testing_loader_B
    def __get_val_loader__(self):
        return None, None
    
    def __get_info__(self): # TODO: MODIFY FOR MULTI TASKING IF THESE VARIABLE USED!
        return (self.L_, self.C_, self.num_class)