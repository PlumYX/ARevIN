import torch
import numpy as np
import random
from exp.exp_main import Exp_Main
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model', type=str, required=True, default='Linear', help='model')
    parser.add_argument('--seed', type=int, default=256, help='seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seg', type=int, default=10, help='prediction plot segments')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # task(forecast)
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='unused fot this model')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=2048, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--rev', action='store_true', default=False, help='RevIN: whether to apply RevIN')
    parser.add_argument('--arev', action='store_true', default=False, help='AdaRevIN: whether to apply AdaRevIN')
    parser.add_argument('--arev_mode', type=str, default='type0', help='AdaRevIN mode: type0 & type1')

    # specific(default)
    parser.add_argument('--individual', action='store_true', default=False, help='Linear: a linear layer for each variate(channel) individually')
    parser.add_argument('--fac_C', action='store_true', default=False, help='TSMixer: whether to apply factorized channel interaction')
    parser.add_argument('--patch_len', type=int, default=16, help='PatchTST: patch length')
    parser.add_argument('--stride', type=int, default=8, help='PatchTST: stride')
    parser.add_argument('--n_heads', type=int, default=8, help='Attention: num of heads')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # for args.model in ['Linear', 'TSMixer', 'PatchTST', 'SCINet']:
    for args.model in ['Linear']:
    # for args.model in ['TSMixer']:
    # for args.model in ['PatchTST']:
    # for args.model in ['SCINet']:
    # for args.model in [args.model]:

        if args.model == 'PatchTST':
            args.d_model = 16
            args.d_ff = 128
            args.n_heads = 4
        elif args.model == 'SCINet':
            args.c_out = 1 if args.features == 'S' else args.c_out
            args.d_model = 64 if args.data_path == 'Electricity.csv' else 128
            args.d_ff = 512  # default=512
            args.dropout = 0.5 if args.data in ['ETTm1', 'ETTm2'] else 0.2

        for i in range(2, 3):
        # for i in ['default']:
            if i == 0:
                args.rev, args.arev = False, False
                args.checkpoints = './checkpoints_{}/'.format(args.seed)
            elif i == 1:
                args.rev, args.arev = True, False
                args.checkpoints = './checkpoints_rev_{}/'.format(args.seed)
            elif i == 2:
                args.rev, args.arev = False, True
                args.checkpoints = './checkpoints_arev_{}/'.format(args.seed)

            args.arev_mode = 'type1' if args.data_path == 'Electricity.csv' else 'type0'
            args.num_workers = 8 if args.features == 'M' else 0

            for args.pred_len in [96, 192, 336, 720]:
            # for args.pred_len in [args.pred_len]:

                fix_seed = args.seed
                random.seed(fix_seed)
                torch.manual_seed(fix_seed)
                np.random.seed(fix_seed)

                print('Args in experiment:')

                basic_config = 'model:{}, is_training:{}, seed:{}'.format(args.model, args.is_training, args.seed)
                print(basic_config)

                data_loader = 'data:{}, features:{}, target:{}, checkpoints:{}, seg:{}, do_predict:{}'.format(
                    args.data, 
                    args.features, 
                    args.target, 
                    args.checkpoints, 
                    args.seg,
                    args.do_predict)
                print(data_loader)

                task = 'seq_len:{}, label_len:{}, pred_len:{}'.format(
                    args.seq_len, 
                    args.label_len, 
                    args.pred_len)
                print(task)

                model = 'enc_in:{}, dec_in:{}, c_out:{}, d_model:{}, d_ff:{}, e_layers:{}, d_layers:{}, dropout:{}'.format(
                    args.enc_in, 
                    args.dec_in, 
                    args.c_out, 
                    args.d_model, 
                    args.d_ff, 
                    args.e_layers, 
                    args.d_layers, 
                    args.dropout)
                print(model)

                rev = 'rev:{}, arev:{}, arev_mode:{}'.format(
                    args.rev, 
                    args.arev, 
                    args.arev_mode)
                print(rev)
                
                specific = 'individual:{}, fac_C:{}, patch_len:{}, n_heads:{}'.format(
                    args.individual, 
                    args.fac_C, 
                    args.patch_len, 
                    args.n_heads)
                print(specific)

                optimization = 'train_epochs:{}, patience:{}, batch_size:{}, learning_rate:{}, lradj:{}, loss:{}, itr:{}'.format(
                    args.train_epochs, 
                    args.patience, 
                    args.batch_size, 
                    args.learning_rate, 
                    args.lradj, 
                    args.loss, 
                    args.itr)
                print(optimization)

                gpu = 'use_amp:{}, use_gpu:{}, gpu:{}, use_multi_gpu:{}, devices:{}, num_workers:{}'.format(
                    args.use_amp, 
                    args.use_gpu, 
                    args.gpu, 
                    args.use_multi_gpu, 
                    args.devices, 
                    args.num_workers)
                print(gpu)

                Exp = Exp_Main

                if args.is_training:
                    for ii in range(args.itr):
                        # setting record of experiments
                        setting = '{}_{}_features_{}_seqlen_{}_predlen_{}_dmodel_{}_dff_{}_elayers_{}_dlayers_{}_itr_{}'.format(
                            args.model,
                            args.data_path[:-4],
                            args.features,
                            args.seq_len,
                            args.pred_len,
                            args.d_model,
                            args.d_ff,
                            args.e_layers,
                            args.d_layers, ii)

                        exp = Exp(args)  # set experiments
                        print('>>>>>>start training: {}>>>>>>'.format(setting))
                        exp.train(setting)

                        time_now = time.time()
                        print('>>>>>>testing: {}<<<<<<'.format(setting))
                        exp.test(setting)
                        print('Inference time: ', time.time() - time_now)

                    if args.do_predict:
                        print('>>>>>>predicting: {}<<<<<<'.format(setting))
                        exp.predict(setting, True)

                    torch.cuda.empty_cache()
                else:
                    ii = 0
                    setting = '{}_{}_features_{}_seqlen_{}_predlen_{}_dmodel_{}_dff_{}_elayers_{}_dlayers_{}_itr_{}'.format(
                        args.model,
                        args.data_path[:-4],
                        args.features,
                        args.seq_len,
                        args.pred_len,
                        args.d_model,
                        args.d_ff,
                        args.e_layers,
                        args.d_layers, ii)

                    exp = Exp(args)  # set experiments
                    print('>>>>>>testing: {}<<<<<<'.format(setting))
                    exp.test(setting, test=1)
                    torch.cuda.empty_cache()
