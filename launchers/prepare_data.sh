# Download images
wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
unzip data_object_image_2.zip

# Download camera calibration
wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
unzip data_object_calib.zip

# Download training labels
wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
unzip data_object_label_2.zip

./launchers/det_precompute.sh ./config/config.py train
./launchers/det_precompute.sh ./config/config.py test

