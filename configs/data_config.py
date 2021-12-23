"""
Configuration file for dataset paths
"""

root_dir = "./dataset"
save_dir = "./saved_models"

# Path to CD2014 images
fr_path = f"{root_dir}/CDnet2014_dataset/{{cat}}/{{vid}}/input/in{{fr_id}}.jpg"
gt_path = f"{root_dir}/CDnet2014_dataset/{{cat}}/{{vid}}/groundtruth/gt{{fr_id}}.png"

# Directory for the selected background frames
selected_frs_200_csv = f"{root_dir}/CDNET2014_selected_frames_200.csv"

# Path to the temp roi file
temp_roi_path = f"{root_dir}/currentFr/{{cat}}/{{vid}}/temporalROI.txt"