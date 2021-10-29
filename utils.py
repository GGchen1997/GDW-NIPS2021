# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import random
import numpy as np
from resnet import ResNet32
from load_corrupted_data import CIFAR10, CIFAR100


def make_onehot_label(targets, num_class):
    return torch.zeros(targets.shape[0], num_class, device=targets.device).scatter_(1, targets.reshape(-1, 1), 1)


def cross_entropy_with_soft_label(logits, targets):
    return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_dataset(args):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='./data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='./data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='./data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='./data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='./data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='./data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100

    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(train_loader.dataset.train_labels) if label == j]

    img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor, args.num_meta)# * num_classes)
    print(img_num_list)
    print(sum(img_num_list))
    im_data = {}
    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        im_data[cls_idx] = img_id_list[img_num:]
        idx_to_del.extend(img_id_list[img_num:])

    print(len(idx_to_del))

    imbalanced_train_dataset = copy.deepcopy(train_data)
    imbalanced_train_dataset.train_labels = np.delete(train_loader.dataset.train_labels, idx_to_del, axis=0)
    imbalanced_train_dataset.train_data = np.delete(train_loader.dataset.train_data, idx_to_del, axis=0)
    imbalanced_train_dataset.clabels = np.delete(train_loader.dataset.clabels, idx_to_del, axis=0)

    imbalanced_train_loader = torch.utils.data.DataLoader(
        imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)

    return imbalanced_train_loader, train_meta_loader, test_loader


def build_model(args):
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(args, optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 90)))  # For WRN-28-10
    lr = args.lr * (1 + np.cos(np.pi * epochs * 1.0 / (args.epochs * 1.0))) / 2.0
    print("lr {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_img_num_per_cls(dataset,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls

