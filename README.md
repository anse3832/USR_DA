# USR_DA
Unofficial pytorch implementation of the paper "Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective"
(https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Unsupervised_Real-World_Super-Resolution_A_Domain_Adaptation_Perspective_ICCV_2021_paper.pdf)

This code doesn't exactly match what the paper describes.
- In the paper, it doesn't provide accurate descriptions of the model (encoder, decoder and discriminator)
- Therefore, I use 5 convolution layers as encoder, RRDB network as decoder, and VGG network as discriminator.

The environmental settings are described below. (I cannot gaurantee if it works on other environments)
- Pytorch=1.7.1+cu110 
- numpy=1.18.3
- cv2=4.2.0
- tqdm=4.45.0

# Train
First, you need to download the NTIRE dataset.
- Download the dataset from this link: https://competitions.codalab.org/competitions/22221
- In the link, downlod all data related to NTIRE20
- After download the dataset, the dataset should be composed as below

![캡처](https://user-images.githubusercontent.com/77471764/139184554-f79e0efb-ef0e-4411-8f11-203b69c6c964.PNG)
- Set the database path in "./opt/option.py" (It is represented as "dir_data")

After those settings, you can run the train code by running "train.py"
- python3 train.py --gpu_id 0 (execution code)
- This code works on single GPU. If you want to train this code in muti-gpu, you need to change this code
- Options are all included in "./opt/option.py". So you should change the variable in "./opt/option.py"

# Inference
First, you need to specify variables in "./opt/option.py"
- dir_test: root folder of test images
- weights: checkpoint file (trained on NTIRE20 dataset)
- results: inference results will be saved on this folder

After those settings, you can run the inference code by running "inference.py"
- python3 inference.py --gpu_id 0 --weights ./weights/epoch20.pth --dir_test /mnt/Dataset/NTIRE20 (execution code)

# Acknolwdgements
We refer to repos below to implement this code.
- official ESRGAN github (https://github.com/xinntao/ESRGAN)
