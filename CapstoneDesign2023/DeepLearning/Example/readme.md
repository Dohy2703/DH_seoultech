* requirements *

pytorch 1.13.1
CUDA 11.7
EasyOCR

* hardware *

pyrealsense D435

* Usage *

predict_230602.py  <-  execute yolov5 predict.py with Pyrealsense. If you press 'q', stop frame and detect word.

change_file_name.py <- Change each name of images and labels in dataset.

replace_dataset_txt_files <- To reduce classes by one

rotate_image_and_txt <- Enable to change the angle of each image by just click the mouse and press any key

predict_add_label_to_txt <- Add label points to txt file in each of dataset(train, valid, test) 
