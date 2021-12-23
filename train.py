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
from utils.data_loader import CD2014Dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FgSegNet v2.0 pyTorch')

    # Input images
    parser.add_argument('--inp_size', metavar='Input Size', dest='inp_size', type=int, default=224,
                        help='Size of the inputs. If equals 0, use the original sized images. '
                             'Assumes square sized input')
    # Optimization
    parser.add_argument('--lr', metavar='Learning Rate', dest='lr', type=float, default=1e-4,
                        help='learning rate of the optimization')
    parser.add_argument('--weight_decay', metavar='weight_decay', dest='weight_decay', type=float, default=1e-2,
                        help='weight decay of the optimization')
    parser.add_argument('--num_epochs', metavar='Number of epochs', dest='num_epochs', type=int, default=30,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', metavar='Minibatch size', dest='batch_size', type=int, default=1,
                        help='Number of samples per minibatch')
    parser.add_argument('--opt', metavar='Optimizer to be used', dest='opt', type=str, default='adam',
                        help='sgd, rmsprop or adam')
    parser.add_argument('--loss', metavar='Loss function to be used', dest='loss', type=str, default='cross-entropy',
                        help='Loss function to be used ce for cross-entropy')

    # Model name
    parser.add_argument('--model_name', metavar='Name of the model for log keeping', dest='model_name',
                        type=str, default='FgSegNet 2.0',
                        help='Name of the model')

    args = parser.parse_args()
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    loss = args.loss
    opt = args.opt
    inp_size = args.inp_size
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

    # Initializations
    dataset_tr = tr_test_config.datasets_tr[5]
    dataset_test = tr_test_config.datasets_test[5]
    save_dir = data_config.save_dir

    mdl_dir = os.path.join(save_dir, fname)

    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    transforms = [
        [aug.Resize((240, 320))],
        [aug.ToTensor()],
        [aug.NormalizeTensor(mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225])]
    ]

    dataloader_tr = CD2014Dataset(
        dataset_tr,
        transforms=transforms
    )
    dataloader_test = CD2014Dataset(
        dataset_test,
        transforms=transforms
    )

    tensorloader_tr = torch.utils.data.DataLoader(
        dataset=dataloader_tr, batch_size=batch_size, shuffle=True, num_workers=1
    )
    tensorloader_test = torch.utils.data.DataLoader(
        dataset=dataloader_test, batch_size=batch_size, shuffle=False, num_workers=1
    )

    start_epoch = 0
    cuda = True

    # load model
    model = FgSegNet(inp_ch=3)

    # setup optimizer
    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    else:
        raise ("opt=%s is not defined, please choose from ('adam', 'sgd')." % opt)

    if loss == "cross-entropy":
        loss_func = losses.cross_entropy

    if cuda:
        model = model.cuda()

    # training
    best_f = 0.0  # For saving the best model
    st = time.time()
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
                torch.save(model, f"{mdl_dir}/model_best.mdl")

            # Save the checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            torch.save(checkpoint, f"{mdl_dir}/checkpoint.pth")
            if (epoch + 1) % 20 == 0:
                torch.save(model, f"{mdl_dir}/model_epoch{epoch + 1}.mdl")

            st = time.time()

    # save the last model
    torch.save(model, f"{mdl_dir}/model_last.mdl")
