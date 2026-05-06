import argparse
import ast

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    """ create parser """
    parser = argparse.ArgumentParser(description='hyper parameters')
    parser.add_argument('--device', type=int, default=0, help="cuda:0")
    parser.add_argument('--root_path', type=str, default="/project_path")
    parser.add_argument('--data_dir', type=str, default="/brain_datasets")

    parser.add_argument('--dataset', default='ABIDE', help='brain dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', default=5, help='repeat time')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--Train_prop', default=0.7)
    parser.add_argument('--Val_prop', default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--base_lr', type=float, default=1e-5)
    parser.add_argument('--target_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--n_clusters', type=lambda s: ast.literal_eval(s), default=[128, 64])
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--s_mlp_layer', type=int, default=2)
    parser.add_argument('--encoder_dim', type=int, default=32)
    parser.add_argument('--sc_heads', type=int, default=8)



    """ The command line reads the parameters """
    return parser.parse_args()



