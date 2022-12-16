import os
import torch, cv2
import numpy as np
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from model.make_model import __num_of_layers


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from matplotlib.backends.backend_pdf import PdfPages
    #
    # x1 = [1, 2, 3, 5, 6, 7]
    # x2 = [1, 6, 14, 15]
    # y1 = [84, 85.07, 85.72, 85.56, 85.96, 85.99]
    # y2 = [84, 85.33, 85.86, 86.24, 85.94, 86]
    # y3 = [84, 86.01, 86.11, 86.17]
    # plt.plot(x1, y1, 's-', color='r', label="Vertical Division")
    # plt.plot(x1, y2, 'o-', color='g', label="Horizontal Division")
    # plt.plot(x2, y3, 'd-', color='b', label="Patch-based Division")
    # plt.xlabel("Number of Parts")
    # plt.ylabel("mAP")
    # plt.legend(loc="best")
    # plt.savefig("map.pdf")
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['TEST.VIS', "True"])
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("pit", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    epochs = cfg.SOLVER.MAX_EPOCHS
    eval_period = cfg.SOLVER.EVAL_PERIOD
    OUTPUT_DIR = cfg.TEST.WEIGHT

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    filename = '0208C6T0044F010'
    with open(filename+'.jpg', 'rb') as f:
        with Image.open(f) as im:
            img = im.convert('RGB')
    img = torch.stack([val_transforms(img)], 0).unsqueeze(0).to("cuda")

    model_path = 'transformer_120.pth'
    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(os.path.join(OUTPUT_DIR, model_path))
    model.to("cuda")
    model.eval()
    attns = model(img, cam_label=[0])

    attn_base, attn_head = attns
    att_base, att_head = torch.stack(attn_base).squeeze(1), torch.stack(attn_head).squeeze(1)

    # Average the attention weights across all heads.
    att_base, att_head = torch.mean(att_base, dim=1), torch.mean(att_head, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att_base, redisual_att_head = \
        torch.eye(att_base.size(1)).to("cuda"), \
        torch.eye(att_head.size(1)).to("cuda")
    aug_att_base, aug_att_head = \
        att_base + residual_att_base, \
        att_head + redisual_att_head
    aug_att_base, aug_att_head = \
        aug_att_base / aug_att_base.sum(dim=-1).unsqueeze(-1), \
        aug_att_head / aug_att_head.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_base.size()).to("cuda")
    joint_attentions[0] = aug_att_base[0]

    for n in range(1, aug_att_base.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_base[n], joint_attentions[n - 1])

    # Attention from the att_base output token to the input space.
    v = joint_attentions[-1]
    img_H, img_W = 21, 10
    mask = v[0, 1:].reshape(img_H, img_W).detach().cpu().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    # # Attention for the att_head output token to the input space
    # cls_token   = joint_attentions[-1][0:1, 0:1 ]
    # patch_token = joint_attentions[-1][0:1,  1:].reshape(img_H, img_W)
    # num_patch = __num_of_layers[cfg.MODEL.LAYER0_DIVISION_TYPE]
    # division_length = patch_token.size(1) // num_patch
    # # tokens = [patch_token[:, i*division_length: (i+1)*division_length] for i in range(num_patch)]
    # # tokens = [torch.cat((cls_token, i), dim=1) for i in tokens]
    # # joint_attentions_head = [torch.matmul(i,j) for i,j in zip(tokens, aug_att_head)]
    #
    # if cfg.MODEL.PYRAMID0_TYPE == 'patch':
    #     if num_patch == 6:
    #         # patch_tokens = [patch_token[m * 7:(m + 1) * 7, n * 5:(n + 1) * 5].reshape(35, 1)
    #         #                for m in range(3) for n in range(2)]
    #         # tokens = [torch.cat((cls_token, i), dim=0) for i in patch_tokens]
    #         # joint_attentions_head = [torch.matmul(i, j) for i, j in zip(aug_att_head, tokens)]
    #         # attns = [i[1:, :].reshape(7, 5) for i in joint_attentions_head]
    #         # attns = torch.cat([torch.cat(
    #         #     [attns[m*2+n] for n in range(2)], dim=1)
    #         #                    for m in range(3)], dim=0)
    #         # attns = attns.reshape(img_H, img_W).detach().cpu().numpy()
    #         # mask = cv2.resize(attns / attns.max(), im.size)[..., np.newaxis]
    #         # result = (mask * im).astype("uint8")
    #
    #         # pos = [[m * 7, (m + 1) * 7, n * 5, (n + 1) * 5] for m in range(3) for n in range(2)]
    #         # pos_store = []
    #         # for p in pos:
    #         #     i,j,m,n = p
    #         #     tmp = []
    #         #     for x in range(i,j):
    #         #         for y in range(m,n):
    #         #             tmp.append(x*10+y+1)
    #         #     pos_store.append(tmp)
    #         # tokens = joint_attentions[-1]
    #         # attentions = []
    #         # for p in pos_store:
    #         #     p_ = [0] + p
    #         #     tmp = [tokens[0][p_]]
    #         #     for p_in in p:
    #         #         tmp.append(tokens[p_in][p_])
    #         #     tmp = torch.stack(tmp, dim=0)
    #         #     attentions.append(tmp)
    #         # joint_attentions_head = [torch.matmul(i, j) for i, j in zip(attentions, aug_att_head)]
    #         # attns = [i[0, 1:].reshape(7, 5) for i in joint_attentions_head]
    #         # attns = torch.cat([torch.cat(
    #         #     [attns[m*2+n] for n in range(2)], dim=1)
    #         #                    for m in range(3)], dim=0)
    #         # attns = attns.reshape(img_H, img_W).detach().cpu().numpy()
    #
    #         att_head_vis = att_head[:,0,1:].reshape(num_patch, 7, 5)
    #         attns = torch.cat([torch.cat(
    #             [att_head_vis[m*2+n] for n in range(2)], dim=1)
    #                            for m in range(3)], dim=0)
    #         p = attns.detach().cpu().numpy()
    #         plt.figure()
    #         plt.imshow(p)
    #         plt.show()
    #
    #
    #
    #     elif num_patch == 14:
    #         attns = [i[:, :, 0, 1:].reshape(1, -1, 3, 5) for i in attns]
    #         attns = torch.cat([torch.cat(
    #             [attns[m * 2 + n] for n in range(2)], dim=3)
    #             for m in range(7)], dim=2)
    #     elif num_patch == 15:
    #         attns = [i[:, :, 0, 1:].reshape(1, -1, 7, 2) for i in attns]
    #         attns = torch.cat([torch.cat(
    #             [attns[m * 2 + n] for n in range(3)], dim=3)
    #             for m in range(5)], dim=2)
    # else:
    #     if cfg.MODEL.PYRAMID0_TYPE == 'horizontal':
    #         feature = feature.reshape(B, N, -1, self.in_planes)
    #     elif cfg.MODEL.PYRAMID0_TYPE == 'vertical':
    #         feature = feature.transpose(-3, -2).reshape(B, N, -1, self.in_planes)
    #     division_length = (features.size(2) - 1) // self.layer_division_num[i]
    #     local_feats = [feature[:, :, m * division_length:(m + 1) * division_length]
    #                    for m in range(self.layer_division_num[i])]

    from matplotlib.backends.backend_pdf import PdfPages
    plt.figure()
    plt.imshow(result)
    # plt.show()
    plt.savefig(filename + '_result.pdf')

    plt.figure()
    plt.imshow(mask)
    # plt.show()
    plt.savefig(filename + '_mask.pdf')

    plt.figure()
    plt.imshow(im)
    # plt.show()
    plt.savefig(filename + '_im.pdf')



