import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Video-based Re-ID',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--log_path', type=str, default='loss.txt')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--resume_validation', type=bool, default=False)
    
    parser.add_argument('--train_txt', help='txt for train dataset')
    parser.add_argument('--train_info', help='npy for train dataset')
    parser.add_argument('--test_txt', help='txt for test dataset')
    parser.add_argument('--train_batch', help='batch size for training')
    parser.add_argument('--test_info', help='npy for test dataset')
    parser.add_argument('--query_info', help='npy for test dataset')
    
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_step_size', type=int, default=50, help='step size of lr')
    
    parser.add_argument('--class_per_batch', type=int, default=8)
    parser.add_argument('--track_per_class', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=6)
    parser.add_argument('--test_batch', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=1)
    
    parser.add_argument("--gpu_id", default='0,1,2')
    parser.add_argument('--eval_freq', type=int, default=2)
    parser.add_argument('--eval_only', action='store_true', default=False)
    
    parser.add_argument('--smem_size', type=int, default=10)
    parser.add_argument('--smem_margin', type=float, default=0.3)
    parser.add_argument('--tmem_size', type=int, default=5)
    parser.add_argument('--tmem_margin', type=float, default=0.3)
    parser.add_argument('--pin_memory', type=bool, default=False)

    parser.add_argument('--mot', type=bool, default=False)

    return parser.parse_args()