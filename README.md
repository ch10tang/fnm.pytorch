# Face Normalization Model
A PyTorch implementation of [Unsupervised Face Normalization with Extreme Pose and Expression in the Wild](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qian_Unsupervised_Face_Normalization_With_Extreme_Pose_and_Expression_in_the_CVPR_2019_paper.pdf) from the paper by Qian, Yichen and Deng, Weihong and Hu, Jiani.

Here are some examples made by fnm.pytorch.
![Alt text](./imgs/Samples.png)



Pre-requisites
-- 
- python3
- Install [Pytorch](https://pytorch.org/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=%2A%2ALP+-+TM+-+General+-+HV+-+TW&utm_adgroup=Install+PyTorch&utm_keyword=%2Binstall%20%2Bpytorch&utm_offering=AI&utm_Product=PyTorch&gclid=Cj0KCQjw1Iv0BRDaARIsAGTWD1uxAZX565HEO1i5eJJ9OE_mshYp7PJ6JBaVNUqZUln93a37cKlhSjUaAppiEALw_wcB) following the website. (or install w/ pip install torch torchvision)
- numpy (install w/ pip install numpy)
- pillow (install w/ pip install pillow)
- matplotlib (install w/ pip install matplotlib)
- tensorboardX (install w/ pip install tensorboardX)
- pandas (install w/ pip install pandas)
- scipy (install w/ pip install scipy)

Datasets
--
- Download face dataset such as CAISA-WebFace, VGGFace2, and MS-Celeb-1M
- All face images are normalized to 250x250 according to landmarks. According to the five facial points, please follow the align protocol in [LightCNN](https://github.com/AlfredXiangWu/LightCNN).

Procedure 
--
1. Colone the Repository. 
2. Download the face expert model from [VGGFace2 Github](https://github.com/ox-vgg/vgg_face2), and put the models in **/Pretrained/VGGFace2/** directory. 


To-do list
--
- [x] Released the training code. 
- [ ] Released the evaluation code.
- [ ] Upload the trained model

