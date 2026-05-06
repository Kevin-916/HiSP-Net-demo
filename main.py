import torch
import datetime
import numpy as np
from data_utils import load_data, init_stratified_dataloader
from train_test import train, val_test
from utils import hyper_para_load, count_param, fix_seed, initialize_logger
from parse import get_args
import os
from model.HiSP_Net import SpectralClustering





def run(args, dataset):
    dataloaders = init_stratified_dataloader(args, *dataset)
    train_loader, val_loader, test_loader = \
        dataloaders["train_dataloader"], dataloaders["val_dataloader"], dataloaders["test_dataloader"]

    # (node_sz, timeseries_sz, node_feature_sz, layers, dropout,
    #  pooling, cluster_num) = hyper_para_load(dataset=dataset, args=args)
    hyper_para_load(dataset=dataset, args=args)

    # model define and load
    model = SpectralClustering(args=args,
                               features_num=args.node_feature_sz,
                               node_num=args.node_sz,
                               n_clusters=args.n_clusters,
                               activation=args.activation,
                               dropout=args.dropout,)
    total_parameters = count_param(model)
    print("Total number of parameters: {}".format(total_parameters))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    logger = initialize_logger()

    epoch_train_loss_list, epoch_train_acc_list = [], []
    epoch_val_roc_list, epoch_val_acc_list, epoch_val_loss_list = [], [], []
    epoch_test_roc_list, epoch_test_acc_list = [], []
    epoch_test_sen_list, epoch_test_spec_list = [], []

    for epoch in range(args.epochs):
        result_train = train(model=model, optimizer=optimizer, args=args, train_loader=train_loader, epoch=epoch)
        result_val_test = val_test(model=model, args=args, val_loader=val_loader, test_loader=test_loader)

        logger.info(" | ".join([
            f'Epoch[{epoch}/{args.epochs}]',
            f'Train Loss:{result_train["train_loss"]: .3f}',
            f'Train Accuracy:{result_train["train_acc"]: .4f}',
            f'Val Loss:{result_val_test["val_loss"]:.3f}',
            f'Val Accuracy:{result_val_test["val_acc"]:.4f}',
            f'Val AUC:{result_val_test["val_roc"]:.4f}',
            f'Test Accuracy:{result_val_test["test_acc"]: .4f}',
            f'Test AUC:{result_val_test["test_roc"]:.4f}',
            f'Test Sen:{result_val_test["test_sensitivity"]:.4f}',
            f'Test Spec:{result_val_test["test_specificity"]:.4f}'
        ]))

        epoch_train_loss_list.append(result_train["train_loss"])
        epoch_train_acc_list.append(result_train["train_acc"])
        epoch_val_loss_list.append(result_val_test['val_loss'])
        epoch_val_acc_list.append(result_val_test["val_acc"])
        epoch_val_roc_list.append(result_val_test['val_roc'])
        epoch_test_roc_list.append(result_val_test['test_roc'])
        epoch_test_acc_list.append(result_val_test['test_acc'])
        epoch_test_sen_list.append(result_val_test['test_sensitivity'])
        epoch_test_spec_list.append(result_val_test['test_specificity'])

    index_max = epoch_val_loss_list.index(min(epoch_val_loss_list))
    # index_max = epoch_val_roc_list.index(max(epoch_val_roc_list))
    return epoch_test_acc_list[index_max], epoch_test_roc_list[index_max], epoch_test_sen_list[index_max], epoch_test_spec_list[index_max]


def main(args):
    # fix_seed(args.seed)

    # load dataset
    dataset = load_data(args)

    runs = args.runs
    run_acc_list, run_roc_list = [], []
    run_sen_list, run_spec_list = [], []
    for i in range(runs):
        print(f'run: {i} start')
        acc, roc, sen, spec = run(args, dataset)
        print(f'run: {i} is over')
        run_acc_list.append(acc)
        run_roc_list.append(roc)
        run_sen_list.append(sen)
        run_spec_list.append(spec)

    acc_mean, acc_std = np.mean(run_acc_list), np.std(run_acc_list)
    roc_mean, roc_std = np.mean(run_roc_list), np.std(run_roc_list)
    sen_mean, sen_std = np.mean(run_sen_list), np.std(run_sen_list)
    spec_mean, spec_std = np.mean(run_spec_list), np.std(run_spec_list)
    print("After ", args.runs, "runs on ", args.dataset, "!")
    print("roc_auc ± std: {:.2f}% ± {:.2f}     ".format(roc_mean * 100, roc_std * 100),
          "mean ± std: {:.2f}% ± {:.2f}".format(acc_mean * 100, acc_std * 100))


    ##################################################
    import pandas as pd
    import os

    result_dir = os.path.join(args.root_path, "result")
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = os.path.join(result_dir, f"{args.dataset}.csv")

    new_row = pd.DataFrame([{
        "acc_mean": round(acc_mean * 100, 2),
        "acc_std": round(acc_std * 100, 2),
        "roc_mean": round(roc_mean * 100, 2),
        "roc_std": round(roc_std * 100, 2),
        "sen_mean": round(sen_mean * 100, 2),
        "sen_std": round(sen_std * 100, 2),
        "spec_mean": round(spec_mean * 100, 2),
        "spec_std": round(spec_std * 100, 2),
        "seed": args.seed,
        "runs": args.runs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_lr": args.base_lr,
        "target_lr": args.target_lr,
        "wd": args.weight_decay,
        "activation": args.activation,
        "dropout": args.dropout,
        "n_clusters": args.n_clusters,
        "alpha": args.alpha,
        "s_mlp_layer": args.s_mlp_layer,
        "sc_heads": args.sc_heads,
    }])

    if os.path.exists(result_file_path):
        new_row.to_csv(result_file_path, mode="a", header=False, index=False)
    else:
        new_row.to_csv(result_file_path, mode="w", header=True, index=False)

    print(f"Results saved to '{result_file_path}'")

   ###################################################################

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)