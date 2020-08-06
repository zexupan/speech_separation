import argparse
import torch
from utils import *
import os
from dpaNet import dpaNet
from solver import Solver


def main(args):
    if args.distributed:
        torch.manual_seed(0)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
    # Model
    model = dpaNet(args.N, args.L, args.B, args.H, args.K, args.R,
                       args.C)

    if (args.distributed and args.local_rank ==0) or args.distributed == False:
        print("started on " + args.log_name + '\n')
        print(args)
        print("\nTotal number of parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
        print(model)
        
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr_dataset = AudioDataset(args.train_dir, args.batch_size,
                              sample_rate=args.sample_rate,
                              segment=args.segment)
    tr_sampler = DistributedSampler(
                            tr_dataset,
                            num_replicas=args.world_size,
                            rank=args.local_rank) if args.distributed else None
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle = (tr_sampler is None),
                                num_workers=args.num_workers,
                                sampler=tr_sampler)

    cv_dataset = AudioDataset(args.valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    cv_sampler = DistributedSampler(
                            cv_dataset,
                            num_replicas=args.world_size,
                            rank=args.local_rank) if args.distributed else None
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                shuffle = (cv_sampler is None),
                                num_workers=args.num_workers,
                                sampler=cv_sampler)

    tt_dataset = AudioDataset(args.test_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    tt_sampler = DistributedSampler(
                            tt_dataset,
                            num_replicas=args.world_size,
                            rank=args.local_rank) if args.distributed else None
    tt_loader = AudioDataLoader(tt_dataset, batch_size=1,
                                shuffle = (tt_sampler is None),
                                num_workers=args.num_workers,
                                sampler=tt_sampler)

    args.train_sampler=tr_sampler

    solver = Solver(args=args,
                model = model,
                optimizer = optimizer,
                train_data = tr_loader,
                validation_data = cv_loader,
                test_data = tt_loader) 
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Conv-tasnet")

    # Dataloader
    parser.add_argument('--train_dir', type=str, default='/home/panzexu/workspace/speech_separation/data/tr',
                    help='directory including mix.json, s1.json and s2.json')
    parser.add_argument('--valid_dir', type=str, default='/home/panzexu/workspace/speech_separation/data/cv',
                    help='directory including mix.json, s1.json and s2.json')
    parser.add_argument('--test_dir', type=str, default='/home/panzexu/workspace/speech_separation/data/tt',
                    help='directory including mix.json, s1.json and s2.json')
    parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
    parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
    parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')

    # Training
    parser.add_argument('--batch_size', default=3, type=int,
                        help='Batch size')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')   
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of maximum epochs')

    # Model hyperparameters
    parser.add_argument('--L', default=2, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of input channels')
    parser.add_argument('--B', default=64, type=int,
                        help='Number of output channels')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')
    parser.add_argument('--H', default=128, type=int,
                        help='Number of hidden size in rnn')
    parser.add_argument('--K', default=256, type=int,
                        help='Number of chunk size')
    parser.add_argument('--R', default=6, type=int,
                        help='Number of layers')

    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Init learning rate')
    parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')


    # Log and Visulization
    parser.add_argument('--log_name', type=str, default=None,
                        help='the name of the log')
    parser.add_argument('--use_tensorboard', type=int, default=0,
                        help='Whether to use use_tensorboard')
    parser.add_argument('--continue_from', type=str, default='',
                        help='Whether to use use_tensorboard')

    # Distributed training
    parser.add_argument('--opt-level', default='O0', type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--patch_torch_functions', type=str, default=None)

    args = parser.parse_args()

    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    main(args)
