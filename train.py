"""
Train script
"""

import argparse
import os
import time
import torch
import torch.optim as optim
import configs.data_config as data_config
import configs.dataset_config as tr_test_config

from models.model import FgSegNet
from utils import losses
from utils import augmentations as aug
from tensorboardX import SummaryWriter
from utils.data_loader import Cd2014Dataset

DEBUG = False


def print_debug(s):
    if DEBUG:
        print(s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FgSegNet v2.0 pyTorch')

    # Input images
    parser.add_argument('--inp_size', metavar='Input Size', dest='inp_size', type=int, default=224,
                        help='Size of the inputs. If equals 0, use the original sized images. '
                             'Assumes square sized input')

    # Training options
    parser.add_argument('--tr_strategy', metavar='Training strategy', dest='tr_str', type=int, default=0,
                        help='0: Video-optimized, 1:Video-agnostic ')

    # Optimization
    parser.add_argument('--lr', metavar='Learning Rate', dest='lr', type=float, default=1e-4,
                        help='learning rate of the optimization')
    parser.add_argument('--weight_decay', metavar='weight_decay', dest='weight_decay', type=float, default=1e-4,
                        help='weight decay of the optimization')
    parser.add_argument('--num_epochs', metavar='Number of epochs', dest='num_epochs', type=int, default=20,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', metavar='Minibatch size', dest='batch_size', type=int, default=4,
                        help='Number of samples per minibatch')
    parser.add_argument('--opt', metavar='Optimizer to be used', dest='opt', type=str, default='adam',
                        help='sgd, rmsprop or adam')
    parser.add_argument('--loss', metavar='Loss function to be used', dest='loss', type=str,
                        default='jaccard',
                        help='Loss function, cross-entropy, jaccard or weighted_crossentropy')

    # Cross-validation
    # You can select more than one train-test split, specify the id's of them
    parser.add_argument('--set_number', metavar='Which training-test split to use from config file', dest='set_number',
                        type=int, default=[0], help='Training and test videos will be selected based on the set number')

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

    # Intialize Tensorboard
    writer = SummaryWriter(f"tb_runs/{fname}")
    print("Initialized TB")

    if torch.cuda.is_available():
        print("GPU is here:)")

    transforms = [
        [aug.Resize((240, 320))],
        [aug.ToTensor()],
        [aug.NormalizeTensor(mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225])]
    ]

    save_dir = data_config.save_dir
    mdl_dir = os.path.join(save_dir, fname)

    if tr_strategy == 0:
        mdl_dir_model_weight = os.path.join(mdl_dir, "video_agnostic")
    elif tr_strategy == 1:
        mdl_dir_model_weight = os.path.join(mdl_dir, "video_optimized")

    saved_weight_ext = "video_agnostic"

    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    if not os.path.exists(mdl_dir_model_weight):
        os.makedirs(mdl_dir_model_weight)

    start_epoch = 0
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


    for (dataset_tr, dataset_test) in zip(dataset_tr_list, dataset_test_list):
        dataloader_tr = Cd2014Dataset(
            dataset_tr,
            transforms=transforms,
            use_selected=200
        )
        dataloader_test = Cd2014Dataset(
            dataset_test,
            transforms=transforms,
            use_selected=200
        )

        tensorloader_tr = torch.utils.data.DataLoader(
            dataset=dataloader_tr, batch_size=batch_size, shuffle=True, num_workers=1
        )
        tensorloader_test = torch.utils.data.DataLoader(
            dataset=dataloader_test, batch_size=batch_size, shuffle=False, num_workers=1
        )

        # load model
        model = FgSegNet(inp_ch=3)

        # setup optimizer
        if opt == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif opt == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ("opt=%s is not defined, please choose from ('adam', 'sgd', 'RMSprop')." % opt)

        if loss == "cross-entropy":
            loss_func = losses.cross_entropy
        elif loss == "jaccard":
            loss_func = losses.jaccard_loss
        elif loss == "weighted_crossentropy":
            loss_func = losses.weighted_crossentropy

        # Print model's state_dict
        print_debug("Model's state_dict:")
        for param_tensor in model.state_dict():
            print_debug(param_tensor + "\t" + str(model.state_dict()[param_tensor].size()))

        # Freeze layers
        for layer in model.frozenLayers:
            for param in layer.parameters():
                param.requires_grad = False

        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        if not os.path.isfile(f"{save_dir}/vgg16-397923af.pth"):
            torch.hub.load_state_dict_from_url(vgg_url, model_dir=mdl_dir,
                                               map_location=None, progress=True, check_hash=False, file_name=None)

        # Load VGG-16 weights
        vgg16_weights = torch.load(f"{save_dir}/vgg16-397923af.pth")
        mapped_weights = {}
        for layer_count, (k_vgg, k_segnet) in enumerate(zip(vgg16_weights.keys(), model.state_dict().keys())):
            # Last layer of VGG16 is not used in encoder part of the model
            if layer_count == 20:
                break
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print_debug("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            model.load_state_dict(mapped_weights)
            print_debug("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


        if cuda:
            model = model.cuda()

        if tr_strategy == 1:
            key, value = list(dataset_tr.items())[0]
            saved_weight_ext = key + '_' + value[0]
        # training
        best_f = 0.0  # For saving the best model
        st = time.time()
        print("Video category/file name is : ", saved_weight_ext)
        for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
            for phase, tensorloader in [("Train", tensorloader_tr), ("Test", tensorloader_test)]:
                running_loss, running_acc, running_f = 0.0, 0.0, 0.0
                if phase == "Train":
                    optimizer.zero_grad()
                for i, data in enumerate(tensorloader):
                    if phase == "Train":
                        model.train()
                    else:
                        model.eval()

                    if phase == "Train":
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data[0].float(), data[1].float()
                        if cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        outputs = model(inputs)
                        labels_1d, outputs_1d = losses.getValid(labels, outputs)
                        loss = loss_func(labels_1d, outputs_1d)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        with torch.no_grad():
                            # get the inputs; data is a list of [inputs, labels]
                            inputs, labels = data[0].float(), data[1].float()
                            if cuda:
                                inputs, labels = inputs.cuda(), labels.cuda()
                            outputs = model(inputs)
                            labels_1d, outputs_1d = losses.getValid(labels, outputs)
                            loss = loss_func(labels_1d, outputs_1d)

                    # print statistics
                    running_loss += loss.item()
                    running_acc += losses.acc(labels_1d, outputs_1d).item()
                    running_f += losses.f_score(labels_1d, outputs_1d).item()

                    del inputs, labels, outputs, labels_1d, outputs_1d
                    if (i + 1) % 10000 == 0:  # print every 2000 mini-batches
                        print("::%s::[%d, %5d] loss: %.1f, acc: %.3f, f_score: %.3f" %
                              (phase, epoch + 1, i + 1,
                               running_loss / (i + 1), running_acc / (i + 1), running_f / (i + 1)))

                epoch_loss = running_loss / len(tensorloader)
                epoch_acc = running_acc / len(tensorloader)
                epoch_f = running_f / len(tensorloader)

                current_lr = lr
                print("::%s:: Epoch %d loss: %.1f, acc: %.3f, f_score: %.3f, lr x 1000: %.4f, elapsed time: %s" \
                      % (phase, epoch + 1, epoch_loss, epoch_acc, epoch_f, current_lr * 1000, (time.time() - st)))

                writer.add_scalar(f"{phase}/loss", epoch_loss, epoch)
                writer.add_scalar(f"{phase}/acc", epoch_acc, epoch)
                writer.add_scalar(f"{phase}/f", epoch_f, epoch)

                if phase.startswith("Test"):
                    best_f = epoch_f
                    torch.save(model, f"{mdl_dir_model_weight}/model_best.mdl")

                # Save the checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }

                torch.save(checkpoint, f"{mdl_dir_model_weight}/checkpoint.pth")
                if (epoch + 1) % 20 == 0:
                    torch.save(model, f"{mdl_dir_model_weight}/model_epoch{epoch + 1}.mdl")

                st = time.time()

        # save the last model
        torch.save(model, f"{mdl_dir_model_weight}/model_{saved_weight_ext}_last.mdl")
