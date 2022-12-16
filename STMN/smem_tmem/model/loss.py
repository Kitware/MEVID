import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.autograd import Variable
import parser
args = parser.parse_args()
import sys
sys.path.append('..')
from util.loss import TripletLoss

class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

        self.criterion_trip = TripletLoss('soft', True)
        self.criterion_id = nn.CrossEntropyLoss().cuda()
    def forward(self, input, labels):
        output = {}

        pool_val_s = torch.mean(
            input['val_s'].reshape(input['val_s'].shape[0]//args.seq_len, args.seq_len, -1), dim=1)
        pool_val_s_cls = torch.mean(
            input['val_s_cls'].reshape(input['val_s_cls'].shape[0]//args.seq_len, args.seq_len, -1), dim=1)

        output['trip'] = self.criterion_trip(pool_val_s, labels, dis_func='eu') \
                                + self.criterion_trip(input['val_t'], labels, dis_func='eu')
        output['track_id'] = self.criterion_id(pool_val_s_cls, labels) \
                                + self.criterion_id(input['val_t_cls'], labels)
        return output
