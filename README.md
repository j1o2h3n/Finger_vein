# Finger_vein
Simple realization of finger vein skeleton extraction

## About

Extract the vein skeleton structure of the input finger vein image.

<p align="center">
  <img width="768" height="243" src=./Fig/Overview.png>
</p>



## Algorithm Code

The algorithm is implemented using traditional digital image processing methods and a relatively simple way to get relatively rough results.

This project uses C++ language to write algorithm code, mainly calls the OpenCV4 library, and uses Visual Studio 2017 as the IDE.

This is the code I completed in March 2018. At that time, I used the OpenCV3 library. In 2019, I updated to the new version of the OpenCV4 library.

For the skeleton extracted by the code, the subsequent tasks such as image recognition and classification can be performed by simple digital image processing methods. I did not upload the subsequent implementation code to this GitHub project.

About this project, you can load this project into the Visual Studio IDE to run the program.

Note: Before running the program, please change the part of the code about reading and writing file locations.

<p align="center">
  <img width="810" height="540" src=./Fig/run.png>
</p>



## Experiment

After running the program, the images obtained through our various image processing steps will be displayed, clearly showing each step of the algorithm.

The algorithm is divided into three steps as a whole. First, the ROI area is extracted by preprocessing, then image enhancement is performed to obtain a binary image, and finally denoising filtering is performed to obtain the final result.

<p align="center">
  <img width="897" height="714" src=./Fig/Process.png>
</p>


