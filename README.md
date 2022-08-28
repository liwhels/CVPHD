# CVPHD
High-Resolution Refocusing for Defocused ISAR Images by Complex-valued Pix2pixHD Network
# Abstract
Inverse synthetic aperture radar (ISAR) is an effective detection method for targets. However, for the maneuvering targets, the Doppler frequency induced by an arbitrary scatterer on the target is time-varying, which will cause defocus on ISAR images, and bring difficulties for the further recognition process. It is hard for traditional methods to well refocus all positions on the target well. In recent years, generative adversarial networks (GAN) achieves great success in image translation. However, the current refocusing models ignore the information of high-order terms containing in the relationship between real parts and imaginary parts of the data. To this end, an end-to-end refocusing network, named Complex-valued Pix2pixHD (CVPHD) is proposed to learn the mapping from defocus to focus, which utilizes complex-valued (CV) ISAR images as input. A complex-valued instance normalization layer is applied to mine the deep relationship between the complex parts by calculating the covariance of them and accelerate the training. Subsequently, an innovative adaptively weighted loss function is put forward to improve the overall refocusing effect. Finally, the proposed CV-P2PHD is tested with the simulated and real dataset, and both can get well-refocused results. The results of comparative experiments show that the refocusing error can be reduced if extending the pix2pixHD network to the CV domain and the performance of CVPHD surpasses other autofocus methods in the refocusing effects.   
![img](https://github.com/yhx-hit/cv_gnn/blob/main/heart.gif)
# Dataset
The self-built refocusing dataset of defocused ISAR images can be downloaded from  
https://pan.baidu.com/s/1VX2uMjW6x0KJHJRztoFqZQ?pwd=0623  
The dataset exists in the .mat format.  
# Read dataset
The ISAR image data is in the form of complex-valued. We can get the content by:  
```
  import h5py
  input_dict = h5py.File(dataset)
  img = input_dict['s3']
  img_real = img['real'].astype(np.float32)
  img_imag = img['imag'].astype(np.float32)
