import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import sys
import numpy as np

sys.path.extend(['../'])
import time
import argparse
from Dataset import adni2
from sklearn import metrics
from Models import CrossKTnet
import yaml


def print_log(string, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        string = "[ " + localtime + ' ] ' + string
    with open('{}/{}_train_log.txt'.format(args['work_dir'], args['dataset']), 'a') as f:
        print(string, file=f)


def log_configuration(args, num_train, num_valid, configs):
    print_log('------------------GPU initialization！ -----------------------------')

    print_log('CUDA_VISIBLE_DEVICES:{}'.format(args['gpu']))
    print_log('****************dataset details********************')
    print_log('dataset:{}'.format(args['dataset']))
    print_log('Samples for train = {}  Samples for valid = {} '.format(num_train, num_valid))

    print_log('*****************train setting********************')
    print_log('train epoch={}  '.format(args['end_epoch']))
    print_log('batch_size={}  '.format(args['batch_size']))
    print_log('lr={} momentum={}'.format(args['lr'], args['momentum']))

    print_log('***************model details********************')
    print_log('sparsity_alpha:{}'.format(args['sparsity_alpha']))
    print_log('kernel_size:{}'.format(args['kernel_size']))
    print_log('config:{}'.format(configs))


def cacu_metric(output, y):
    predict = torch.argmax(output, dim=-1)
    ACC = torch.sum(predict == y)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y = y.cpu()
    predict = predict.cpu()
    for i in range(len(y)):
        if y[i] == 1 and predict[i] == 1:
            TP += 1
        elif y[i] == 0 and predict[i] == 0:
            TN += 1
        elif y[i] == 0 and predict[i] == 1:
            FP += 1
        elif y[i] == 1 and predict[i] == 0:
            FN += 1

    return ACC / len(y), TP, TN, FP, FN, TP / (TP + FN), TN / (TN + FP)


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    if args['seed']:
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])
        '''random.seed(args['seed)
        np.random.seed(args['seed)
        torch.cuda.manual_seed_all(args['seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True'''

    # Save the optimal values of 5-folds
    all_best_ACC = np.zeros(5)
    all_best_SEN = np.zeros(5)
    all_best_SPE = np.zeros(5)
    all_best_AUC = np.zeros(5)

    for k in args['k_folds']:

        train_data = adni2(data=args['prefix'], split=k, mode='train')
        valid_data = adni2(data=args['prefix'], split=k, mode='test')
        one_sample, ___ = train_data.__getitem__(1)
        num_frame = one_sample.shape[-3]
        num_point = one_sample.shape[-2]
        num_class = valid_data.get_num_class()

        num_train = len(train_data)
        num_valid = len(valid_data)

        train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True,
                                  drop_last=True, pin_memory=True)  # worker_init_fn=np.random.seed(args['seed)

        valid_loader = DataLoader(dataset=valid_data, batch_size=num_valid, shuffle=False,
                                  drop_last=False, pin_memory=True)

        torch.cuda.set_device(args['gpu'])
        if os.path.exists("./check_points") is False:
            os.makedirs('./check_points')  # Save weights
        if os.path.exists(args['work_dir']) is False:
            os.makedirs(args['work_dir'])  # Save weights


        log_configuration(args, num_train, num_valid, args['config_128'])

        print_log('num_frame={} num_point={}'.format(num_frame, num_point))

        print_log('---------------------{}split------------------'.format(k))

        My_mode = CrossKTnet(sparsity_alpha=args['sparsity_alpha'], num_subset=args['num_subset'], num_frame=num_frame,
                             num_point=num_point,
                             kernel_size=args['kernel_size'], use_pes=args['use_pes'], num_class=num_class,
                             config=args['config_128'])

        My_mode = My_mode.cuda(args['device'])
        My_mode = torch.nn.DataParallel(My_mode, device_ids=[args['gpu']])

        if args['pre_trained']:  # Load the pre-training weight
            print_log('loading   weights：{}'.format(args['weights_path']))
            weights_dict = torch.load(args['weights_path'], map_location=lambda storage, loc: storage)
            My_mode.load_state_dict(weights_dict['state_dict'])

        optimizer = torch.optim.SGD(My_mode.parameters(), lr=args['lr'], momentum=args['momentum'], nesterov=True,
                                    weight_decay=args['lr_decay_rate'])
        loss_F = nn.CrossEntropyLoss().to(args['device'])

        Best_ACC = 0

        for epoch in range(args['start_epoch'], args['end_epoch']):
            My_mode.train()  # Train
            if epoch < args['warm_up_epoch']:
                lr = args['lr'] * (epoch + 1) / args['warm_up_epoch']
            else:
                lr = args['lr'] * (args['lr_decay_rate'] ** np.sum(epoch >= np.array(args['step'])))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            total_loss = 0
            train_ACC = 0
            for i, data in enumerate(train_loader):
                x, target = data
                x = x.cuda(args['device'], non_blocking=True)
                target = target.cuda(args['device'], non_blocking=True)
                output = My_mode(x)

                loss = loss_F(output, target)
                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_ACC += torch.sum((torch.argmax(output, dim=-1) == target))

            My_mode.eval()  # Test
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    x, target = data
                    x = x.cuda(args['device'], non_blocking=True)
                    target = target.cuda(args['device'], non_blocking=True)
                    out_put = My_mode(x)

                    ACC, TP, TN, FP, FN, SEN, SPE = cacu_metric(out_put, target)

                    fpr, tpr, thresholds = metrics.roc_curve(target.cpu().detach().numpy(),
                                                             out_put[:, 1].cpu().detach().numpy(), pos_label=1)
                    auc = metrics.auc(fpr, tpr)

            if ACC > Best_ACC:
                Best_ACC = ACC
                all_best_SEN[k - 1] = SEN
                all_best_SPE[k - 1] = SPE
                all_best_AUC[k - 1] = auc
                all_best_ACC[k - 1] = Best_ACC

            print_log('split:{} Epoch: {}  loss:{:.5f}  train_ACC:{:.5f} test_ACC:{:.5f} Best_ACC:{:.5f} '.format(k,
                                                                                                                  epoch,
                                                                                                                  total_loss / len(
                                                                                                                      train_loader),
                                                                                                                  train_ACC / num_train,
                                                                                                                  ACC,
                                                                                                                  Best_ACC))
            print('split:{} Epoch: {}  loss:{:.5f}  train_ACC:{:.5f} test_ACC:{:.5f} Best_ACC:{:.5f} '.format(k,
                                                                                                              epoch,
                                                                                                              total_loss / len(
                                                                                                                  train_loader),
                                                                                                              train_ACC / num_train,
                                                                                                              ACC,
                                                                                                              Best_ACC))

            print_log('TP:{}  TN:{}  FP:{} FN:{} '.format(TP, TN, FP, FN))
            print_log('SEN: {:.5f}  SPE: {:.5f} auc:{:.5f}'.format(SEN, SPE, auc))

            if ((epoch + 1) % int(args['save_freq']) == 0):  # save model
                file_name = os.path.join(
                    './check_points/{}_split{}_epoch_{}.pth'.format(args['dataset'], k, epoch))  # checkpoint_dir
                torch.save({
                    'epoch': epoch,
                    'state_dict': My_mode.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

    with open('Final_results.txt', 'a') as f:
        localtime = time.asctime(time.localtime(time.time()))

        print(args['dataset'], file=f)
        print(
            "[ " + localtime + ' ] ' + 'SEN: {} Average: {:.5f} std: {:.5f}'.format(all_best_SEN, np.mean(all_best_SEN),(np.std(all_best_SEN))), file=f)
        print(
            "[ " + localtime + ' ] ' + 'SPE: {} Average: {:.5f} std: {:.5f}'.format(all_best_SPE, np.mean(all_best_SPE), (np.std(all_best_SPE))), file=f)
        print(
            "[ " + localtime + ' ] ' + 'AUC: {} Average: {:.5f} std: {:.5f}'.format(all_best_AUC, np.mean(all_best_AUC),(np.std(all_best_AUC))), file=f)


if __name__ == '__main__':
    with open('./config.yaml', 'r', encoding='utf8') as file:  #
        args = yaml.safe_load(file)
    main(args)








