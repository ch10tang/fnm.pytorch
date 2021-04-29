# Face Normalization Model
A PyTorch implementation of [Unsupervised Face Normalization with Extreme Pose and Expression in the Wild](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qian_Unsupervised_Face_Normalization_With_Extreme_Pose_and_Expression_in_the_CVPR_2019_paper.pdf) from the paper by Qian, Yichen and Deng, Weihong and Hu, Jiani.

Here are some examples made by fnm.pytorch.
![Alt text](./imgs/Samples.png)



Pre-requisites
-- 
- python3
- CUDA 9.0 or higher
- Install [Pytorch](https://pytorch.org/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=%2A%2ALP+-+TM+-+General+-+HV+-+TW&utm_adgroup=Install+PyTorch&utm_keyword=%2Binstall%20%2Bpytorch&utm_offering=AI&utm_Product=PyTorch&gclid=Cj0KCQjw1Iv0BRDaARIsAGTWD1uxAZX565HEO1i5eJJ9OE_mshYp7PJ6JBaVNUqZUln93a37cKlhSjUaAppiEALw_wcB) following the website. (or install w/ pip install torch torchvision)
- numpy 
- pillow 
- matplotlib 
- tensorboardX 
- pandas 
- scipy

Datasets
--
- Download face dataset such as CAISA-WebFace, VGGFace2, and MS-Celeb-1M as source set, and you can use any constrained (in-the-house) dataset as normal set.
- All face images are normalized to 250x250 according to landmarks. According to the five facial points, please follow the align protocol in [LightCNN](https://github.com/AlfredXiangWu/LightCNN). I also provide the crop code (MTCNN) which as shown below.

Training and Inference 
--
1. Colone the Repository to preserve Directory Strcuture. 
2. Download the [face expert model](https://drive.google.com/drive/folders/1V7oMdPm2gmoBXKLsHrlzD0Gx2yAyk8qZ?usp=sharing), and put the model in **/Pretrained/VGGFace2/** directory. 
3. Change the directory to **/FaceAlignment/** (*cd FaceAlignment*), and crop and align the input face images by running:

    ```python face_align.py```
4. Train the face normalization model by running:

    ```python main.py -front-list {} -profile-list {}```
5. I also provide a simple test code, which can help to generate the normalized face and extract the features:

    ```python main.py -generate -gen-list {} -snapshot {your trained model}```

Note that, you need to define the csv files of source/normal/generate data roots during training/testing.


To-do list
--
- [x] Released the training code. 
- [x] Released the evaluation code.


