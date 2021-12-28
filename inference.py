import os
import argparse
import torch
import csv
import configs.data_config as data_config
import configs.dataset_config as tr_test_config
from utils import augmentations as aug
from utils.data_loader import Cd2014Dataset
from utils import losses
from models.model import FgSegNet
from utils.eval_utils import logVideos

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FgSegNet v2.0 pyTorch')

    # Input images
    parser.add_argument('--inp_size', metavar='Input Size', dest='inp_size', type=int, default=224,
                        help='Size of the inputs. If equals 0, use the original sized images. '
                             'Assumes square sized input')

    # Training options
    parser.add_argument('--tr_strategy', metavar='Training strategy', dest='tr_str', type=int, default=1,
                        help='0: Video-optimized, 1:Video-agnostic ')

    # Optimization
    parser.add_argument('--lr', metavar='Learning Rate', dest='lr', type=float, default=1e-4,
                        help='learning rate of the optimization')
    parser.add_argument('--weight_decay', metavar='weight_decay', dest='weight_decay', type=float, default=1e-4,
                        help='weight decay of the optimization')
    parser.add_argument('--num_epochs', metavar='Number of epochs', dest='num_epochs', type=int, default=10,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', metavar='Minibatch size', dest='batch_size', type=int, default=1,
                        help='Number of samples per minibatch')
    parser.add_argument('--opt', metavar='Optimizer to be used', dest='opt', type=str, default='adam',
                        help='sgd, rmsprop or adam')
    parser.add_argument('--loss', metavar='Loss function to be used', dest='loss', type=str,
                        default='jaccard',
                        help='Loss function, cross-entropy, jaccard or weighted_crossentropy')

    # Cross-validation
    # You can select more than one train-test split, specify the id's of them
    parser.add_argument('--set_number', metavar='Which training-test split to use from config file', dest='set_number',
                        type=int, default=[5], help='Training and test videos will be selected based on the set number')

    # Model name
    parser.add_argument('--model_name', metavar='Name of the model for log keeping', dest='model_name',
                        type=str, default='FgSegNet 2.0',
                        help='Name of the model')

    args = parser.parse_args()
    lr = args.lr
    tr_strategy = args.tr_str
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    loss = args.loss
    opt = args.opt
    inp_size = args.inp_size
    dataset_number = args.set_number
    fname = args.model_name
    if inp_size == 0:
        inp_size = None
    else:
        inp_size = (inp_size, inp_size)

    save_dir = data_config.save_dir
    mdl_dir = os.path.join(save_dir, fname)
    csv_path = "./log_video_agnostic.csv"

    if tr_strategy == 0:
        mdl_dir_model_weight = os.path.join(mdl_dir, "video_agnostic")
        csv_path = "./log_video_agnostic.csv"
    elif tr_strategy == 1:
        mdl_dir_model_weight = os.path.join(mdl_dir, "video_optimized")
        csv_path = "./log_video_optimized.csv"

    cuda = True

    # Initializations
    dataset_tr_list = []
    dataset_test_list = []

    if tr_strategy == 1:
        dataset_tr_list = tr_test_config.datasets_tr[0]
        dataset_test_list = tr_test_config.datasets_test[0]
        video_optimized_dataset_tr = []
        for key in dataset_tr_list:
            for value in dataset_tr_list[key]:
                video_optimized_dataset_tr.append({key: [value]})

        dataset_tr_list = video_optimized_dataset_tr
        dataset_test_list = video_optimized_dataset_tr

    elif tr_strategy == 0:
        for number in dataset_number:
            dataset_tr_list.append(tr_test_config.datasets_tr[number])
            dataset_test_list.append(tr_test_config.datasets_test[number])

    # Locations of each video in the CSV file
    csv_header2loc = data_config.csv_header2loc

    category_row = [0] * csv_header2loc['len']
    category_row[0] = 'category'

    scene_row = [0] * csv_header2loc['len']
    scene_row[0] = 'scene'

    metric_row = [0] * csv_header2loc['len']
    metric_row[0] = 'metric'

    for cat, vids in tr_test_config.datasets_tr[0].items():
        for vid in vids:
            category_row[csv_header2loc[vid]] = cat
            category_row[csv_header2loc[vid] + 1] = cat
            category_row[csv_header2loc[vid] + 2] = cat

            scene_row[csv_header2loc[vid]] = vid
            scene_row[csv_header2loc[vid] + 1] = vid
            scene_row[csv_header2loc[vid] + 2] = vid

            metric_row[csv_header2loc[vid]] = 'FNR'
            metric_row[csv_header2loc[vid] + 1] = 'Prec'
            metric_row[csv_header2loc[vid] + 2] = 'F-score'

    with open(csv_path, mode='w', newline="") as log_file:
        employee_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(category_row)
        employee_writer.writerow(scene_row)
        employee_writer.writerow(metric_row)

    # Evaluation on test videos
    for (dataset_tr, dataset_test) in zip(dataset_tr_list, dataset_test_list):

        if tr_strategy == 1:
            key, value = list(dataset_tr.items())[0]
            saved_weight_ext = key + '_' + value[0]

        elif tr_strategy == 0:
            saved_weight_ext = "video_agnostic"

        model = torch.load(f"{mdl_dir_model_weight}/model_{saved_weight_ext}_last.mdl").cuda()

        # Inference selected train-test splits in the config file
        if tr_strategy == 0:
            for idx in dataset_number:
                dataset_test = tr_test_config.datasets_test[idx]
                logVideos(
                    dataset_test,
                    model,
                    fname,
                    csv_path,
                    save_vid=1,
                    debug=0
                )

        if tr_strategy == 1:
            logVideos(
                dataset_test,
                model,
                fname,
                csv_path,
                save_vid=1,
                debug=0
            )

    print(f"Saved results to {csv_path}")
