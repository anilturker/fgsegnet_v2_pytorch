import csv
import torch
from utils.losses import getValid
from utils.data_loader import Cd2014Dataset
from utils.visualize import tensor2double
from utils import augmentations as aug
import cv2
import configs.data_config as data_config
import os
import numpy as np

# Locations of each video in the CSV file
csv_header2loc = data_config.csv_header2loc

def evalVideo(cat, vid, model, eps=1e-5,
              save_vid=False, save_outputs="", model_name="", debug=False, use_selected=False, multiplier=16):
    """ Evalautes the trained model on all ROI frames of cat/vid
    Args:
        :cat (string):                  Category
        :video (string):                Video
        :model (torch model):           Trained PyTorch model
        :eps (float):                   A small multiplier for making the operations easier
        :save_vid (boolean):            Boolean for saving the output as a video
        :save_outputs (str):            Folder path to save the outputs If = "" do not save
        :model_name (string):           Name of the model for logging. Important when save_vid=True
        :debug (boolean):               Use for quick debugging
    """

    transforms = [
        [aug.Resize((240, 320))],
        [aug.ToTensor()],
        [aug.NormalizeTensor(mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225])]
    ]

    dataloader = Cd2014Dataset({cat:[vid]}, transforms=transforms,
                              use_selected=use_selected, multiplier=0)
    tensorloader = torch.utils.data.DataLoader(dataset=dataloader,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1)

    if save_vid:
        im = next(iter(dataloader))[0][0]
        h, w = im.shape
        if model_name.endswith("_manualBG"):
            model_name = model_name[:-9]
        if model_name.endswith("_autoBG"):
            model_name = model_name[:-7]

        video_result_path = data_config.video_result_dir
        if not os.path.exists(video_result_path):
            os.makedirs(video_result_path)
        vid_path = os.path.join(video_result_path, f"{cat}_{vid}.mp4")
        print(vid_path)
        vid = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (3*w+20, h))

    if save_outputs:
        output_path = os.path.join(data_config.save_dir, "outputs", save_outputs)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, "results")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, cat)):
            os.makedirs(os.path.join(output_path, cat))
        if not os.path.exists(os.path.join(output_path, cat, vid)):
            os.makedirs(os.path.join(output_path, cat, vid))

    model.eval() # Evaluation mode
    tp, fp, fn = 0, 0, 0

    for i, data in enumerate(tensorloader):

        if debug and i >= 100:
            break
        if (i+1) % 1000 == 0:
            print("%d/%d" %(i+1, len(tensorloader)))
        input, label = data[0].float(), data[1].float()

        input, label = input.cuda(), label.cuda()
        _, _, h, w = input.shape
        right_pad, bottom_pad = -w % multiplier, -h % multiplier
        zeropad = torch.nn.ZeroPad2d((0, right_pad, 0, bottom_pad))

        input = zeropad(input)
        output = model(input)
        
        output = output[:, :, :h, :w]
        label_1d, output_1d = getValid(label, output)

        if save_vid:
            input_np = tensor2double(input)
            label_np = label.cpu().detach().numpy()[0, 0, :, :]
            output_np = output.cpu().detach().numpy()[0, 0, :, :]

            vid_fr = np.ones((h, 3*w+20, 3))*0.5
            #print(vid_fr.shape, input_np.shape, label_np.shape, output_np.shape)
            vid_fr[:, :w, :] = input_np[:, :, -3:]

            for k in range(3):
                vid_fr[:, w+10:2*w+10, k] = label_np
                vid_fr[:, 2*w+20:, k] = output_np

            vid.write((vid_fr[:, :, ::-1]*255).astype(np.uint8))

        if save_outputs:
            output_np = output.cpu().detach().numpy()[0, 0, :, :]
            output_np = (output_np > 0.5) * 1
            h, w = output_np.shape
            output_fr = np.ones((h, w, 3))
            for k in range(3):
                output_fr[:, :, k] = output_np
            fname = os.path.join(output_path, cat, vid, f"bin{str(i+1).zfill(6)}.png")
            cv2.imwrite(fname, (output_fr*255).astype(np.uint8))
            
        tp += eps * torch.sum(label_1d * output_1d).item()
        fp += eps * torch.sum((1-label_1d) * output_1d).item()
        fn += eps * torch.sum(label_1d * (1-output_1d)).item()
        del input, label, output, label_1d, output_1d

    # Calculate the statistics
    prec = tp / (tp + fp) if tp + fp > 0 else float('nan')
    recall = tp / (tp + fn) if tp + fn > 0 else float('nan')
    f_score = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else float('nan')
    if save_vid:
        vid.release()

    return 1-recall, prec, f_score

def logVideos(dataset, model, model_name, csv_path, eps=1e-5,
              save_vid=False, save_outputs="", debug=False):
    """ Evaluate the videos given in dataset and log them to a csv file
    Args:
        :dataset (dict):                Dictionary of dataset. Keys are the categories (string),
                                        values are the arrays of video names (strings).
        :model (torch model):           Trained PyTorch model
        :model_name (string):           Name of the model for logging
        :csv_path (string):             Path to the CSV file
        :empty_bg (boolean):            Boolean for using the empty background frame
        :recent_bg (boolean):           Boolean for using the recent background frame
        :segmentation_ch (boolean):     Boolean for using the segmentation maps
        :eps (float):                   A small multiplier for making the operations easier
        :save_vid (boolean):             Boolean for saving the output as a video
        :debug (boolean):               Use for quick debugging
    """

    new_row = [0] * csv_header2loc['len']
    new_row[0] = model_name

    for cat, vids in dataset.items():
        for vid in vids:
            print(vid)
            fnr, prec, f_score = evalVideo(cat, vid, model, eps=eps, save_vid=save_vid,
                                           save_outputs=save_outputs, model_name=model_name, debug=debug)

            new_row[csv_header2loc[vid]] = fnr
            new_row[csv_header2loc[vid]+1] = prec
            new_row[csv_header2loc[vid]+2] = f_score


    with open(csv_path, mode='a', newline="") as log_file:
        employee_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(new_row)

    print('Done!!!')
