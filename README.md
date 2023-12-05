# Digital-Net
A digital net is a term to describe the the purpose of the software provided within these scripts. The purpose is to act as a 'net' that can identify objects within a video as they come across the screen.

NOTE
This implementation is specifically designed for videos that are considered the "dolly zoom" or "vertigo affect". These are videos where the objects in the screen are coming towards the camera, starting far in the middle and appearing to move to the outsides of the video as the objects come closer to the camera.


# How to use

## Step 1
Create a python env evironment and install the needed packages: Ultralytics for Yolo, Byte tracking, Supervision.
Take a look at the install page [here](Install)

## Step 2
Create a dataset of images from the source video(s) that need to be tested.
    This can be done by taking screenshots throughout the video and of related videos. The larger this dataset of images, the better the results will be. 
    [FFmpeg](https://github.com/FFmpeg/FFmpeg) is a great tool for automating this process with large datasets. Otherwise, most video editing software have a screenshot option while watching the videos.

## Step 3
Label the images from the dataset before training. Find a labeling software that can give you [Yolov8 format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format) of the images. In the end you need a .txt file for each image in the dataset that contains YOLO coordinates of each object in that image.
A free one that can be used online for small datasets is [Roboflow](roboflow.com).

Create two folders, train and val. Split the labeled images up 80% in the train, and 20% in the val. Keep each images corresponding .txt file with it in that folder. Move both of these folders into a datasets folder that is within project directoy.

## Step 4 
Train a Yolo model with the labeled dataset. 
Revisit [this yolov8 page](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format) to review how to create a correct .yaml file with your data. After this is done, a model can be trained at the command line with the following prompts. Yolo includes models that such as the 'yolov8s' -small- model. Larger datasets can use the medium or large models.

`yolo detect train data=example.yaml model=yolov8s.pt epochs=10 batch=8 imgsz=640`

You can rerun this with new parameters until the optimal model is found.

## Step 5
Run the Digital_Net script with the needed arguments to receive the finished video.
`python3 Digital_Net.py -v /path/to/originalVideo -m /path/to/trainedYoloModel -o OutputFileName`



