# -*- coding: utf-8 -*-
import argparse
import torch.nn.parallel
import torch.optim
import torch.utils.data
from utils import *
from resnet import VNet

parser = argparse.ArgumentParser(description='PyTorch GDW Training')
# data setting
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--imb_factor', type=float, default=1.0)
parser.add_argument('--num_meta', type=int, default=1000)
# training setting
parser.add_argument('--epochs', default=80, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--hidden_dim', '--hidden-dim', default=100, type=int,
                    help='hidden dim (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--outer_lr', '--outer-learning-rate', default=100, type=float,
                    help='initial learning rate')
parser.add_argument('--clip', default=0.2, type=float, help='clip')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.add_argument('--mode', '-mode', type=str, default='gdw', help='You can compare meta-weight-net with gdw by setting this value')

parser.set_defaults(augment=True)
args = parser.parse_args()

set_seed(args.seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100

T = args.epochs
L_EPS = 1e-6
H_EPS = 1e6
print()
print(args)


def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    print('\n Current Epoch: %d' % epoch)
    train_loss = 0
    meta_loss = 0
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets, true_targets, indexs) in enumerate(train_loader):
        model.train()

        # 0. data preparation
        inputs, targets, true_targets = inputs.to(device), targets.to(device), true_targets.to(device)

        # 1. compute foward v_lambda
        meta_model = build_model(args).to(device)
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(inputs)
        one_hot_label = torch.zeros(targets.shape[0], outputs.shape[1]).to(device).scatter_(1,targets.reshape(-1,1),1).to(device)
        p = F.softmax(outputs, dim=1)
        p.data = torch.clamp(p.data, 0, 0.999)
        label_wise_error = p - one_hot_label
        if args.mode == 'mw-net':
            cost = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))
            v_lambda = vnet(cost_v.data)
        elif args.mode == 'gdw':
            cost = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))
            v_lambda_t = vnet(cost_v.data).view(-1, 1)
            v_lambda = v_lambda_t.clone()
            for c in range(num_classes-1):
                v_lambda = torch.cat([v_lambda, v_lambda_t.clone()], dim=1)
        # zero-mean constraint
        if args.mode == 'gdw':
            index = torch.arange(v_lambda.size(0))
            v_lambda[index, targets] = (torch.sum(v_lambda * p.data, dim=1) - v_lambda[index, targets] * p.data[
                index, targets]) / (1 - p.data[index, targets])

        # 2. build the connection
        if len(v_lambda.size()) == 1:
            meta_model.zero_grad()
            loss = F.cross_entropy(outputs, targets, reduction='none')
            loss = torch.mean(loss * v_lambda)
            grad2 = torch.autograd.grad(loss, (meta_model.params()), create_graph=True)
        else:
            meta_model.zero_grad()
            grad1 = label_wise_error * v_lambda / label_wise_error.size()[0]
            grad_sum = torch.sum(grad1)
            grad_sum.backward(retain_graph=True)
            meta_model.zero_grad()
            grad2 = torch.autograd.grad(outputs, (meta_model.params()), grad_outputs=grad1, create_graph=True)
            del grad1
        meta_lr = args.lr * ( 1 + np.cos(np.pi*epoch*1.0/(T*1.0)) )/2.0
        meta_model.update_params(lr_inner=meta_lr, source_params=grad2)
        del grad2

        # 3. update vnet
        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)
        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]
        if args.mode == 'gdw':
            v_lambda.retain_grad()
            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()
            grad_info = v_lambda.grad.data
            deno = torch.sum(torch.abs(grad_info))
            update = torch.clamp(args.outer_lr*grad_info / deno, -args.clip, args.clip)
            v_lambda.data = v_lambda.data - update.data
        elif args.mode == 'mw-net':
            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()
        meta_loss += l_g_meta.item()

        #4. compute new foward w_new
        meta_model = build_model(args).to(device)
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(inputs)
        one_hot_label = torch.zeros(targets.shape[0], outputs.shape[1]).to(device).scatter_(1, targets.reshape(-1, 1),1).to(device)
        with torch.no_grad():
            if args.mode == 'mw-net':
                w_new = vnet(cost_v.data)
            elif args.mode == 'gdw':
                w_new = torch.clamp(v_lambda.data, L_EPS, H_EPS)
        outputs = model(inputs)
        p = F.softmax(outputs, dim=1)
        p.data = torch.clamp(p.data, 0, 0.999)
        label_wise_error = p - one_hot_label
        #zero-mean constraint
        if args.mode == 'gdw':
            index = torch.arange(w_new.size(0))
            w_new[index, targets] = (torch.sum(w_new * p.data, dim=1) - w_new[index, targets] * p.data[
                        index, targets]) / (1 - p.data[index, targets])
        prec_real_train = accuracy(outputs.data, true_targets.data, topk=(1,))[0]

        #5. update model param
        if len(w_new.size()) == 1:
            optimizer_model.zero_grad()
            loss = F.cross_entropy(outputs, targets, reduction='none')
            loss = torch.mean(loss * w_new.data)
            loss.backward()
        else:
            optimizer_model.zero_grad()
            loss = F.cross_entropy(outputs, targets)
            grad1 = label_wise_error.data * w_new.data / label_wise_error.size()[0]
            outputs.backward(grad1)
        optimizer_model.step()
        train_loss += loss.item()
        if batch_idx % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f\t' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_real_train, prec_meta))


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy


def main():
    train_loader, train_meta_loader, test_loader = build_dataset(args)
    # create model
    model = build_model(args)
    
    vnet = VNet(1, args.hidden_dim, 1).to(device)
    
    optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
    
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3,
                                      weight_decay=1e-4)
    best_acc = 0
    history_acc = []
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer_model, epoch)
        train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch)
        test_acc = test(model=model, test_loader=test_loader)
        history_acc.append(test_acc)
        if test_acc >= best_acc:
            best_acc = test_acc
    print('final accuracy: {}'.format(np.mean(history_acc[-5:])))


if __name__ == '__main__':
    main()
