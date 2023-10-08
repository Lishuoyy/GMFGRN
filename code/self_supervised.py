import argparse
import os
import random
import string
import time

import numpy as np
import torch
from data import ScRNASeqData
from model import *
from utils import (
    MetricLogger,
    torch_net_info,
    torch_total_param_num,
)
import logging


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self._act = nn.ReLU()
        self.encoder = GCMCLayer(
            args.rating_vals,
            args.src_in_units,
            args.dst_in_units,
            args.gcn_agg_units,
            args.gcn_out_units,
            args.gcn_dropout,
            device=args.device,
        )
        self.decoder = Decoder(dropout_rate=args.gcn_dropout)

    def forward(self, enc_graph, dec_graph, ufeat, ifeat):
        gene_out, cell_out = self.encoder(enc_graph, ufeat, ifeat)
        pred_ratings = self.decoder(dec_graph, gene_out, cell_out)
        return pred_ratings, gene_out, cell_out


def train(args):
    print(args)

    dataset = ScRNASeqData(
        args.data_name,
        args.device,
        istime=args.is_time,
        ish5=args.is_h5,
        path=args.data_path,
        rank_all=args.class_rating,
    )
    print("Loading data finished ...\n")

    args.src_in_units = dataset.gene_feature_shape[1]
    args.dst_in_units = dataset.cell_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values
    print("src_in_units: ", args.src_in_units)
    print("dst_in_units: ", args.dst_in_units)
    print("rating_vals: ", args.rating_vals)

    ### build the net
    net = Net(args=args)
    net = net.to(args.device)

    mse_loss = nn.MSELoss()
    learning_rate = args.train_lr
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate
    )
    print("Loading network finished ...\n")

    ### perpare training data
    train_gt_labels = dataset.train_labels
    train_gt_ratings = dataset.train_truths
    print('train_gt_labels', len(train_gt_labels))
    print('train_gt_ratings', len(train_gt_ratings))

    ### prepare the logger
    train_loss_logger = MetricLogger(
        ["iter", "mse_loss"],
        ["%d", "%.4f"],
        os.path.join(args.save_dir, "train_loss%d.csv" % args.save_id),
    )

    ### declare the loss information
    best_train_mse = np.inf
    best_iter = -1
    count_loss = 0

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(args.device)

    print("Start training ...")
    dur = []
    start_time = time.time()
    for iter_idx in range(1, args.train_max_iter + 1):
        if iter_idx > 3:
            t0 = time.time()
        net.train()
        pred_ratings, gene_out, cell_out = net(
            dataset.train_enc_graph,
            dataset.train_dec_graph,
            dataset.gene_feature,
            dataset.cell_feature,
        )
        pred_ratings = pred_ratings.squeeze()
        loss = mse_loss(pred_ratings, train_gt_ratings)
        count_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
        optimizer.step()

        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (torch_total_param_num(net)))
            print(
                torch_net_info(
                    net,
                    save_path=os.path.join(
                        args.save_dir, "net%d.txt" % args.save_id
                    ),
                )
            )

        if iter_idx == 1 or iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(
                iter=iter_idx,
                mse_loss=count_loss / (iter_idx + 1),
            )

            logging_str = (
                "Iter={}, mse_loss={:.4f}, time={:.4f}".format(
                    iter_idx,
                    count_loss / iter_idx,
                    np.average(dur),
                )
            )
            if (count_loss / iter_idx) < best_train_mse:
                best_iter = iter_idx
                best_train_mse = (count_loss / iter_idx)
                torch.save(net.state_dict(), args.emb_path + 'best_train_model_v' + '.pth')
                np.save(args.emb_path + 'gene_embedding.npy', gene_out.cpu().detach().numpy())
                np.save(args.emb_path + 'cell_embedding.npy', cell_out.cpu().detach().numpy())

        if iter_idx % args.train_log_interval == 0 or iter_idx == 1:
            print(logging_str)
    print(
        "Best Iter Idx={}, Best MSE={:.4f}".format(
            best_iter, best_train_mse
        )
    )
    logging.info("best_iter_idx: {}, best_MSE: {}".format(best_iter, best_train_mse))
    logging.info("Cost Time: {}".format((time.time() - start_time)))
    train_loss_logger.close()
    print('Cost Time: ', time.time() - start_time)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument(
        "--device",
        default="0",
        type=int,
        help="Running device. E.g `--device 0`, if using cpu, set `--device -1`",
    )
    parser.add_argument("--save_dir", type=str, help="The saving directory")
    parser.add_argument("--save_id", type=int, help="The saving log id")
    parser.add_argument(
        "--data_name",
        default="mHSC_E",
        type=str,
        help="The dataset name: mHSC_E, mHSC_GM, mHSC_L, timeData/hesc1...",
    )
    parser.add_argument(
        "--data_path",
        default="../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv",
        type=str,
    )
    # ../Beeline-master/BeelineData/500_hESC/ExpressionData.csv
    # data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv
    # data_evaluation/bonemarrow/bone_marrow_cell.h5
    # data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/
    parser.add_argument("--is_time", default=False, action="store_true")
    parser.add_argument("--is_h5", default=False, action="store_true")
    parser.add_argument("--gcn_dropout", type=float, default=0.0)
    parser.add_argument("--gcn_agg_units", type=int, default=3840)
    parser.add_argument("--gcn_out_units", type=int, default=256)

    parser.add_argument("--train_max_iter", type=int, default=20000)

    parser.add_argument("--train_log_interval", type=int, default=100)

    parser.add_argument("--train_grad_clip", type=float, default=0.1)
    parser.add_argument("--train_lr", type=float, default=0.01)

    parser.add_argument("--class_rating", type=int, default=15)
    args = parser.parse_args()
    args.device = (
        th.device(args.device) if args.device >= 0 else th.device("cpu")
    )

    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = (
                args.data_name
                + "_"
                + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=2)
        )
        )
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == "__main__":
    args = config()
    path = "../embeddings/" + args.data_name + '/'
    args.emb_path = path
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + "train.log", level=logging.INFO)
    args_dict = args.__dict__
    for k, v in args_dict.items():
        logging.info("{}: {}".format(k, v))
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    train(args)
