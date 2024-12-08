# (We are updating the information and adjusting the pages on this code! If you want to provide some good papers, please send us on the issues! Hope that we can provide some intreseting works for the omnidirectional vision!)

# &diams; Deep Learning for Omnidirectional Vision: A Survey and New Perspectives

<!-- <div align=center>
<img src="https://github.com/VLISLAB/360-DL-Survey/blob/main/Images/main_page/survey.png" width="1000" height="450">
</div> -->

**Referenced paper** : [Deep Learning for Omnidirectional Vision: A Survey and New Perspectives](https://arxiv.org/abs/2205.10468)

## Table of Content
### &hearts; Introduction

&nbsp; &nbsp; &nbsp; &nbsp; Omnidirectional image (ODI) data is captured with a  360°×180°  field-of-view and omnidirectional vision has attracted booming attention due to its more advantageous performance in numerous applications. Our survey presents a systematic and comprehensive review and analysis of the recent progress in Deep Learning methods for omnidirectional vision. 
 
&nbsp; &nbsp; &nbsp; &nbsp; **Especially, we create this open-source repository to provide a taxonomy of all the mentioned works and code links in the survey. We will keep updating our open-source repository with new works in this area and hope it can shed light on future research and build a community for researchers on omnidirectional vision.**
### &hearts; Background
#### **Acquisition**
An ideal 360&deg; camera can capture lights falling on the focal point from all directions, making the projection plane a whole spherical surface. In practice, most 360&deg; cameras can not achieve it, which excludes top and bottom regions due to dead angles.

### &hearts; Acquision

- **Photometric consistency for dual fisheye cameras (2020)** [paper](https://ieeexplore.ieee.org/iel7/9184803/9190635/09190784.pdf)

- **Generating a full spherical view by modeling the relation between two fisheye images (2024)** [paper](https://link.springer.com/content/pdf/10.1007/s00371-024-03293-7.pdf)

- **Attentive deep stitching and quality assessment for 360&deg omni-directional images (2019)** [paper](https://ieeexplore.ieee.org/iel7/4200690/5418892/08903278.pdf)

- **Image stitching for dual fisheye cameras (2018)** [paper](https://ieeexplore.ieee.org/iel7/8436606/8451009/08451333.pdf)

- **Efficient and accurate stitching for 360&deg dual-fisheye images and videos (2017)** [paper](https://ieeexplore.ieee.org/iel7/83/4358840/09633229.pdf)

- **Automatic 360 mono-stereo panorama generation using a cost-effective multi-camera system (2020)** [paper](https://www.mdpi.com/1424-8220/20/11/3097/pdf)

- **360 panorama generation using drone mounted fisheye cameras (2019)** [paper](https://ieeexplore.ieee.org/iel7/8656627/8661828/08661954.pdf)

- **Panoramic SLAM from a multiple fisheye camera rig (2020)** [paper](https://www.researchgate.net/profile/Zijie-Qin/publication/337647400_Panoramic_SLAM_from_a_multiple_fisheye_camera_rig/links/5df7b99892851c836482f369/Panoramic-SLAM-from-a-multiple-fisheye-camera-rig.pdf)

- **Review on panoramic imaging and its applications in scene understanding (2022)** [paper](https://ieeexplore.ieee.org/iel7/19/4407674/09927463.pdf)

- **Deep learning on image stitching with multi-viewpoint images: A survey (2023)** [paper](https://link.springer.com/article/10.1007/s11063-023-11226-z)

### &hearts; Projections

- **Spherical light fields. (2014)** [paper](https://www.researchgate.net/profile/Bernd-Krolla/publication/265755272_Spherical_Light_Fields/links/5423feab0cf26120b7a70d36/Spherical-Light-Fields.pdf)

- **3d scene geometry estimation from 360&deg imagery: A survey (2022)** [paper](https://dl.acm.org/doi/pdf/10.1145/3519021)

- **Spherical stereo for the construction of immersive vr environment (2005)** [paper](https://ieeexplore.ieee.org/abstract/document/1492777)

- **3d scene reconstruction from multiple spherical stereo pairs (2013)** [paper](https://link.springer.com/content/pdf/10.1007/s11263-013-0616-1.pdf)

- **Sphorb: A fast and robust binary feature on the sphere (2014)** [paper](https://link.springer.com/content/pdf/10.1007/s11263-014-0787-4.pdf)

- **BiFuse: Monocular 360-degree Depth Estimation via Bi-Projection Fusion (2020)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/BiFuse/Paper.md) [**Code**](https://github.com/Yeh-yu-hsuan/BiFuse)

- **Tangent images for mitigating spherical distortion (2020)** [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Eder_Tangent_Images_for_Mitigating_Spherical_Distortion_CVPR_2020_paper.pdf)

- **Introduction to geometry (1962)** Book.

- **OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/OmniFusion/Paper.md) [**Code**](https://github.com/yuyanli0831/OmniFusion)

- **SpherePHD: Applying CNNs on a Spherical PolyHeDron Representation of 360&deg; Images (2019)**  [Paper](Backbone/SpherePHD/Paper.md) [**Code**](https://github.com/KAIST-vilab/SpherePHD_public) [**Reproduced Code**](https://github.com/keevin60907/SpherePHD)

- **Subjective assessment of 360&deg image projection formats (2020)** [paper](https://ieeexplore.ieee.org/iel7/6287639/6514899/08999504.pdf)

- **Recent advances in omnidirectional video coding for virtual reality: Projection and evaluation (2018)** [paper](https://www.sciencedirect.com/science/article/pii/S0165168418300057)

### &hearts; Datasets.

- **Recognizing scene viewpoint using panoramic place representation (2012)** [paper](https://dspace.mit.edu/bitstream/handle/1721.1/90932/Torralba_Recognizing%20scene.pdf?sequence=1)

- **Matterport3d: Learning from rgb-d data in indoor environments (2017)** [paper](https://arxiv.org/pdf/1709.06158)

- **Joint 2D-3D semantic data for indoor scene understanding (2017)** [paper](https://arxiv.org/abs/1702.01105)

- **Panocontext: A whole-room 3d context model for panoramic scene understanding (2014)** [paper](https://oar.princeton.edu/bitstream/88435/pr1qg23/1/PanoContext.pdf)

- **Generating 360 outdoor panorama dataset with reliable sun position estimation (2018)** [paper](https://dl.acm.org/doi/pdf/10.1145/3283289.3283348)

- **Text2light: Zero-shot text-driven hdr panorama generation (2022)** [paper](https://dl.acm.org/doi/abs/10.1145/3550454.3555447)

- **Learning to predict indoor illumination from a single image (2017)** [paper](https://arxiv.org/pdf/1704.00090)

- **Deep sky modeling for single image outdoor lighting estimation (2019)** [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hold-Geoffroy_Deep_Sky_Modeling_for_Single_Image_Outdoor_Lighting_Estimation_CVPR_2019_paper.pdf)

- **Lau-net: Latitude adaptive upscaling network for omnidirectional image super-resolution (2021)** [paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Deng_LAU-Net_Latitude_Adaptive_Upscaling_Network_for_Omnidirectional_Image_Super-Resolution_CVPR_2021_paper.pdf)

- **Ntire 2023 challenge on 360deg omnidirectional image and video super-resolution: Datasets, methods and results ()** [paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Cao_NTIRE_2023_Challenge_on_360deg_Omnidirectional_Image_and_Video_Super-Resolution_CVPRW_2023_paper.pdf)

- **Testbed for subjective evaluation of omnidirectional visual content (2016)** [paper](https://ieeexplore.ieee.org/iel7/7897355/7906305/07906378.pdf)

- **CVIQD: Subjective quality evaluation of compressed virtual reality images (2017)** [paper](https://ieeexplore.ieee.org/iel7/8267582/8296222/08296923.pdf)

- **Perceptual quality assessment of omnidirectional images (2018)** [paper](https://ieeexplore.ieee.org/iel7/8334884/8350884/08351786.pdf)

- **Spatial attention-based non-reference perceptual quality prediction network for omnidirectional images (2021)** [paper](https://ieeexplore.ieee.org/iel7/9428049/9428068/09428390.pdf)

- **Perceptual Quality Assessment of Omnidirectional Images: A Benchmark and Computational Model (2024)** [paper](https://dl.acm.org/doi/pdf/10.1145/3640344)

- **360-indoor: Towards learning real-world objects in 360deg indoor equirectangular images (2020)** [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Chou_360-Indoor_Towards_Learning_Real-World_Objects_in_360deg_Indoor_Equirectangular_Images_WACV_2020_paper.pdf)

- **Object detection in equirectangular panorama (2018)** [paper](https://ieeexplore.ieee.org/iel7/8527858/8545020/08546070.pdf)

- **Grid based spherical cnn for object detection from panoramic images (2019)** [paper](https://www.mdpi.com/1424-8220/19/11/2622/pdf)

- **Pandora: A panoramic detection dataset for object with orientation (2022)** [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680229-supp.pdf)

- **Spherical criteria for fast and accurate 360 object detection (2020)** [paper](https://ojs.aaai.org/index.php/AAAI/article/view/6995/6849)

- **Spherenet: Learning spherical representations for detection and classification in omnidirectional images (2018)** [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Coors_SphereNet_Learning_Spherical_ECCV_2018_paper.pdf)

- **Capturing omni-range context for omnidirectional segmentation (2021)** [paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Capturing_Omni-Range_Context_for_Omnidirectional_Segmentation_CVPR_2021_paper.pdf)

- **Densepass: Dense panoramic semantic segmentation via unsupervised domain adaptation with attention-augmented context exchange (2021)** [paper](https://arxiv.org/pdf/2108.06383)

- **Behind every domain there is a shift: Adapting distortion-aware vision transformers for panoramic semantic segmentation (2024)** [paper](https://ieeexplore.ieee.org/iel8/34/4359286/10546335.pdf)

- **Orientation-aware semantic segmentation on icosahedron spheres (2019)** [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Orientation-Aware_Semantic_Segmentation_on_Icosahedron_Spheres_ICCV_2019_paper.pdf)

- **MODE: Multi-view Omnidirectional Depth Estimation with 360&deg Cameras (2022)** [paper](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_12)

- **360 depth estimation in the wild-the depth360 dataset and the segfuse network (2022)** [paper](https://ieeexplore.ieee.org/iel7/9756663/9756727/09756738.pdf)

- **Pano3d: A holistic benchmark and a solid baseline for 360 depth estimation (2021)** [paper](https://ieeexplore.ieee.org/iel7/9522011/9522684/09522738.pdf)

- **Self-supervised learning of depth and camera motion from 360&deg videos (2018)** [paper](https://arxiv.org/pdf/1811.05304)

- **Omnidepth: Dense depth estimation for indoors spherical panoramas (2018)** [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/NIKOLAOS_ZIOULIS_OmniDepth_Dense_Depth_ECCV_2018_paper.pdf)

- **A dataset of head and eye movements for 360 degree images (2017)** [paper](https://dl.acm.org/doi/pdf/10.1145/3083187.3083218?casa_token=v9nNfjCe6wQAAAAA:ixsxh0Z56wOAT-zKSc7VfcBu1whltT27aCJek-CWF7AF1zjTDg1wjPvZ105B-4vOqkyEtxlDdum4)

- **Saliency prediction on omnidirectional image with generative adversarial imitation learning (2021)** [paper](https://ieeexplore.ieee.org/iel7/83/9263394/09328187.pdf?casa_token=Uvih0MVxpPoAAAAA:DANzCXLQjlCwnZ41p5dqHd2L8DlKln6J7vEr4sxG7eRyLA7O_LRCJiq4_0nOdXpncaR332o1cfw)

- **Cube padding for weakly-supervised saliency prediction in 360 videos (2018)** [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_Cube_Padding_for_CVPR_2018_paper.pdf)

- **Predicting head movement in panoramic video: A deep reinforcement learning approach (2018)** [paper](https://ieeexplore.ieee.org/iel7/34/8855044/08418756.pdf?casa_token=1aVewpZqotMAAAAA:qSTJLtbrY18pY798NVYnhf-qwuvT4tW12AVF2DJtVXc5N6hA2nDzNhUt0Tb5T3jFce2Ej5O6rsc)

- **Saliency in VR: How do people explore virtual environments? (2018)** [paper](https://ieeexplore.ieee.org/iel7/2945/8315156/08269807.pdf?casa_token=EDd-qL9_0MQAAAAA:FbzbC4qQirjepdMgDoSddgvjUFVB-JaID3EpEMgKLeD4pm-_CcyPd7pyGeiuvQubqcxZHokYypY)

- **Zillow indoor dataset: Annotated floor plans with 360deg panoramas and 3d room layouts (2021)** [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cruz_Zillow_Indoor_Dataset_Annotated_Floor_Plans_With_360deg_Panoramas_and_CVPR_2021_paper.pdf)

- **Deep3dlayout: 3d reconstruction of an indoor layout from a spherical panoramic image (2021)** [paper](https://dl.acm.org/doi/pdf/10.1145/3478513.3480480?casa_token=xcQjKIJZE2oAAAAA:F6TFo_2SMPVDYhTR8AF8n6y5vj1RJnpwJxuF-8He0usqIdIR6AcYPMCvurx1lEcarOcR2TZVCXtW)

- **Gibson env: Real-world perception for embodied agents (2018)** [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xia_Gibson_Env_Real-World_CVPR_2018_paper.pdf)

- **Geometric structure based and regularized depth estimation from 360 indoor imagery (2020)** [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Geometric_Structure_Based_and_Regularized_Depth_Estimation_From_360_Indoor_CVPR_2020_paper.pdf)

- **Panocontext: A whole-room 3d context model for panoramic scene understanding (2014)** [paper](https://oar.princeton.edu/bitstream/88435/pr1qg23/1/PanoContext.pdf)

- **Recognizing scene viewpoint using panoramic place representation (2012)** [paper](https://dspace.mit.edu/bitstream/handle/1721.1/90932/Torralba_Recognizing%20scene.pdf?sequence=1)

- **360+ x: A Panoptic Multi-modal Scene Understanding Dataset (2024)** [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_360x_A_Panoptic_Multi-modal_Scene_Understanding_Dataset_CVPR_2024_paper.pdf)

- **The Replica dataset: A digital replica of indoor spaces (2019)** [paper](https://arxiv.org/pdf/1906.05797)

- **Structured3d: A large photo-realistic dataset for structured 3d modeling (2020)** [paper](https://arxiv.org/pdf/1908.00222)

### &hearts; ODI Representation Learning

- **Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions (2022)** [paper](https://arxiv.org/pdf/2209.13603)

- **PDO-eS2CNNs: Partial differential operator based equivariant spherical CNNs (2021)** [paper](https://ojs.aaai.org/index.php/AAAI/article/download/17154/16961)

- **Spherical CNNs on unstructured grids (2019)** [paper](https://arxiv.org/pdf/1901.02039)

- **Sampling based spherical transformer for 360 degree image classification (2024)** [paper](https://www.sciencedirect.com/science/article/pii/S0957417423023552)

- **HexNet: An Orientation-Aware Deep Learning Framework for Omni-Directional Input (2023)** [paper](https://ieeexplore.ieee.org/iel7/34/4359286/10225707.pdf?casa_token=jZLIV8El0NMAAAAA:FYU6M0v7Cx8GZ0rtTGB4sw3YpjwU6aJCCuOufNt9TNm0Uhw1-YXwOrGqiAhVPQEQSTymvOFVt_I)

- **Deep learning 3d shapes using alt-az anisotropic 2-sphere convolution (2018)** [paper](https://openreview.net/pdf?id=rkeSiiA5Fm)

- **Panoswin: a pano-style swin transformer for panorama understanding (2023)** [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ling_PanoSwin_A_Pano-Style_Swin_Transformer_for_Panorama_Understanding_CVPR_2023_paper.pdf)

- **Learning Spherical Convolution for Fast Features from 360&deg; Imagery (2017)**  [Paper](Backbone/SPHCONV/Paper.md) [**Code**](https://github.com/sammy-su/Spherical-Convolution)

- **Learning SO(3) Equivariant Representations with Spherical CNNs (Point cloud) (2018)**  [Paper](Backbone/SO(3)%20Equivariant%20Representations/Paper.md) [**Code**](https://github.com/daniilidis-group/spherical-cnn)

- **Spherical CNNs (2018)**  [Paper](Backbone/Spherical%20CNNs/Paper.md) [**Code**](https://github.com/jonkhler/s2cnn)

- **SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional Images (2018)**  [Paper](Backbone/SphereNet/Paper.md) [**Code**](https://github.com/BlueHorn07/sphereConv-pytorch)

- **Learning Spherical Convolution for 360&deg Recognition (2021)** [paper](https://ieeexplore.ieee.org/iel7/34/9910243/09541093.pdf?casa_token=_FhgMK6eUXUAAAAA:26UhjVO3Vv2WCWeLREEiIdtlgrJdULTc_-TvgfVF4pfIVihPZ8ozyImLY08SmX3R3CVW4t0wASE)

- **Geometry Aware Convolutional Filters for Omnidirectional Images Representation (Graph-based) (2019)**  [Paper](Backbone/GAfilters/Paper.md) [**Code**](https://github.com/RenataKh/GAfilters)

- **Kernel Transformer Networks for Compact Spherical Convolution (2019)**  [Paper](Backbone/KTN/Paper.md) [Code](Backbone/KTN/Code.md)

- **SpherePHD: Applying CNNs on a Spherical PolyHeDron Representation of 360&deg; Images (2019)**  [Paper](Backbone/SpherePHD/Paper.md) [**Code**](https://github.com/KAIST-vilab/SpherePHD_public) [**Reproduced Code**](https://github.com/keevin60907/SpherePHD)

- **Rotation Equivariant Graph Convolutional Network for Spherical Image Classification (2020)**  [Paper](Backbone/SGCN/Paper.md) [**Code**](https://github.com/QinYang12/SGCN)

- **Deepsphere: A Graph-based Spherical CNN (2020)**  [Paper](Backbone/Deepsphere/Paper.md) [**Code**](https://github.com/deepsphere/deepsphere-pytorch)

- **Equivariant Networks for Pixelized Spheres (2021)**  [Paper](Backbone/Equivariant%20Networks/Paper.md) [**Code**](https://github.com/mshakerinava/Equivariant-Networks-for-Pixelized-Spheres)

- **Equivariance versus Augmentation for Spherical Images (2022)**  [Paper](Backbone/S2CNNseg/Paper.md) [**Code**](https://github.com/JanEGerken/sem_seg_s2cnn)

- **Gauge Equivariant Convolutional Networks and the Icosahedral CNN (2019)**  [Paper](Backbone/Gauge_Equivariant_CNN/Paper.md) [**Code**](https://github.com/VLISLAB/360-DL-Survey/tree/main/Backbone/Gauge_Equivariant_CNNs)

- **Spherical Transformer (2022)**  [Paper](Backbone/Spherical_Transformer/Paper.md) [Code](Backbone/Spherical_Transformer/Code.md)

### &hearts; Optimization Strategies

- **Semi-supervised 360 depth estimation from multiple fisheye cameras with pixel-level selective loss (2022)** [paper](https://ieeexplore.ieee.org/iel7/9745891/9746004/09746232.pdf?casa_token=Y1I3huPcEmgAAAAA:JVXFiRGpg8VTxrG2m9fRKEx4EPDvOppcYJR1g-5fh8Ck670hitG6whkBeIstfrZv-YfYYJwGnck)

- **Unsupervised omnimvs: Efficient omnidirectional depth inference via establishing pseudo-stereo supervision (2023)** [paper](https://ieeexplore.ieee.org/iel7/10341341/10341342/10342332.pdf?casa_token=1oxtDQo7JdIAAAAA:mAERhj_68HisHUYDe3QIxHtd4W8WMokbjOpzsZtIcp9DLW2s3JgeZzgby0nVnIij2dGVsXs5Qp4)

- **Spherical view synthesis for self-supervised 360 depth estimation (2019)** [paper](https://ieeexplore.ieee.org/iel7/8882741/8885410/08885706.pdf?casa_token=Zi9ugJZNYa8AAAAA:uAo1rFHbNv3LbQCEjBN_qk38OPRstXKEtJn-kI4Bh9qD9xXvEVJK1npVMQ3vlqDzA3bXbsQzsUg)

- **Pano-SfMLearner: Self-Supervised multi-task learning of depth and semantics in panoramic videos (2021)** [paper](https://ieeexplore.ieee.org/iel7/97/4358004/09406330.pdf?casa_token=ZaDIco_uk_UAAAAA:N--1GGrXFJXW61hubaDlaGlb25k1G4YdpwP-TOA0EHeAG9iOSy3c5KPyALf3dubrsfFdhdOBIwY)

- **Sslayout360: Semi-supervised indoor layout estimation from 360° panorama (2021)** [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Tran_SSLayout360_Semi-Supervised_Indoor_Layout_Estimation_From_360deg_Panorama_CVPR_2021_paper.pdf?ref=https://githubhelp.com)

- **360fusionnerf: Panoramic neural radiance fields with joint guidance (2023)** [paper](https://ieeexplore.ieee.org/iel7/10341341/10341342/10341346.pdf?casa_token=Bm0pB6beabsAAAAA:m8orZFX8aRNMHCcgZEzY6HIgQAJHOczOhjzeSEcDzMLraf17x0loSbujLOEvGwe0pMBZagXP-Dk)

- **Moving in a 360 world: Synthesizing panoramic parallaxes from a single panorama (2021)** [paper](https://arxiv.org/abs/2106.10859)

- **Enhancement of novel view synthesis using omnidirectional image completion (2022)** [paper](https://arxiv.org/pdf/2203.09957)

- **Learning Omnidirectional Flow in 360 Video via Siamese Representation (2022)** [paper](https://link.springer.com/chapter/10.1007/978-3-031-20074-8_32)

- **Rethinking 360deg Image Visual Attention Modelling With Unsupervised Learning (2021)** [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Djilali_Rethinking_360deg_Image_Visual_Attention_Modelling_With_Unsupervised_Learning._ICCV_2021_paper.pdf)

- **Spherenet: Learning spherical representations for detection and classification in omnidirectional images (2018)** [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Coors_SphereNet_Learning_Spherical_ECCV_2018_paper.pdf)

- **Distortion-aware convolutional filters for dense prediction in panoramic images (2018)** [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Keisuke_Tateno_Distortion-Aware_Convolutional_Filters_ECCV_2018_paper.pdf)

- **Kernel transformer networks for compact spherical convolution (2019)** [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Su_Kernel_Transformer_Networks_for_Compact_Spherical_Convolution_CVPR_2019_paper.pdf)

- **Revisiting optical flow estimation in 360 videos (2021)** [paper](https://ieeexplore.ieee.org/iel7/9411940/9411911/09412035.pdf?casa_token=ameWprmUiuMAAAAA:Ub0fUCfBUWUo6HQ0Qlf4HMKNA1svGmu76S8xRC5VtWBxSFiUlkbCJEARcrgwwfxQWTwnyXlQbZs)

- **Learning Spherical Convolution for 360&dg Recognition (2021)** [paper](https://ieeexplore.ieee.org/iel7/34/9910243/09541093.pdf?casa_token=h4fB8Jopx6sAAAAA:OZYBB0VqKzwKJm0mJSIfla9Y7janoK4wHhUF_yWr-FHqe858lTMlmQKWfeOcE3rFxs-JnTVrU3E)

- **Omniflownet: a perspective neural network adaptation for optical flow estimation in omnidirectional images (2021)** [paper](https://ieeexplore.ieee.org/iel7/9411940/9411911/09412745.pdf?casa_token=2NVe_y7E53UAAAAA:CTLkPJIUaMEWNh7xB-lFSjPj4KwNNUJoFx7_D3GcEv-SQDpo3DK9lIGwAD58DAkXsbJRph5ODIY)

- **PanelNet: Understanding 360 Indoor Environment via Panel Representation (2023)** [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_PanelNet_Understanding_360_Indoor_Environment_via_Panel_Representation_CVPR_2023_paper.pdf)

- **Omnisupervised omnidirectional semantic segmentation (2020)** [paper](https://ieeexplore.ieee.org/iel7/6979/4358928/09204767.pdf?casa_token=8NEM52cqKxgAAAAA:OhsDT5uXwdckKJj0zndKCBVWNSB6zgSYI6VlGJ7aYcXkA5q9JqUuWxyNTP6ph9qa-FNFatRolVI)

- **Both style and distortion matter: Dual-path unsupervised domain adaptation for panoramic semantic segmentation (2023)** [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_Both_Style_and_Distortion_Matter_Dual-Path_Unsupervised_Domain_Adaptation_for_CVPR_2023_paper.pdf)

- **360-attack: Distortion-aware perturbations from perspective-views (2022)** [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_360-Attack_Distortion-Aware_Perturbations_From_Perspective-Views_CVPR_2022_paper.pdf)

- **Capturing omni-range context for omnidirectional segmentation (2021)** [paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Capturing_Omni-Range_Context_for_Omnidirectional_Segmentation_CVPR_2021_paper.pdf)

- **Transfer beyond the field of view: Dense panoramic semantic segmentation via unsupervised domain adaptation (2021)** [paper](https://ieeexplore.ieee.org/iel7/6979/9826234/09599375.pdf?casa_token=gj_Q1zZ2ZxUAAAAA:ZZnduyZ9RMtZ2EaBsHREatGoPKZKnbPdIyRp9QL7-SdOcSjQ6RcMMUkeFOQoIfMi-m-PInP4Pc4)

- **Densepass: Dense panoramic semantic segmentation via unsupervised domain adaptation with attention-augmented context exchange (2021)** [paper](https://arxiv.org/pdf/2108.06383)

- **Bending reality: Distortion-aware transformers for adapting to panoramic semantic segmentation (2022)** [paper](http://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Bending_Reality_Distortion-Aware_Transformers_for_Adapting_to_Panoramic_Semantic_Segmentation_CVPR_2022_paper.pdf)

- **Look at the neighbor: Distortion-aware unsupervised domain adaptation for panoramic semantic segmentation (2023)** [paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Look_at_the_Neighbor_Distortion-aware_Unsupervised_Domain_Adaptation_for_Panoramic_ICCV_2023_paper.pdf)

- **Viewport-based CNN: A multi-task approach for assessing 360&deg video quality (20)** [paper](https://ieeexplore.ieee.org/iel7/34/4359286/09212608.pdf?casa_token=gc-oYI17NvYAAAAA:ZyR3x-jr5q9hHhZLw_2Q6DfQe7MPZulFa1pm4LUG4DDlLY65ONREEiHzE24V0vByQxQ6IngiXMA)

- **Omnidirectional image quality assessment by distortion discrimination assisted multi-stream network (2021)** [paper](https://ieeexplore.ieee.org/iel7/76/4358651/09432940.pdf?casa_token=M8LAmN9ccHEAAAAA:hRKaApOypWXtG44tSy9LhVmDAM23sETYwF1TZ2g3dDBKW5168hQQFyOs4yIy0sVKCyqvyGTsXfg)

- **Deeppanocontext: Panoramic 3d scene understanding with holistic scene context graph and relation-based optimization (2021)** [paper](http://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_DeepPanoContext_Panoramic_3D_Scene_Understanding_With_Holistic_Scene_Context_Graph_ICCV_2021_paper.pdf)

- **PanoContext-Former: Panoramic total scene understanding with a transformer (2024)** [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Dong_PanoContext-Former_Panoramic_Total_Scene_Understanding_with_a_Transformer_CVPR_2024_paper.pdf)

- **MA360: Multi-agent deep reinforcement learning based live 360-degree video streaming on edge (2020)** [paper](https://ieeexplore.ieee.org/iel7/9099125/9102711/09102836.pdf?casa_token=Vivbj28VgfQAAAAA:r9ZnCxxQoX6J5m2Bac_4fozBoBx0biXIBinn_07ZJrGtaBdHdFDkxriloB848PecqXCHfs4zUrE)

- **DRL360: 360-degree video streaming with deep reinforcement learning (2019)** [paper](https://ieeexplore.ieee.org/iel7/8732929/8737365/08737361.pdf?casa_token=LI8qe_KEFpgAAAAA:H17OCyA08OlQd4KyWE88xv2nKCp_ZTPSRCfv-NYzkdpXYmL8mU8SGzPN1zXeTjNpdxhZ3Lw_4SA)

- **Adaptive streaming of 360-degree videos with reinforcement learning (2021)** [paper](http://openaccess.thecvf.com/content/WACV2021/papers/Park_Adaptive_Streaming_of_360-Degree_Videos_With_Reinforcement_Learning_WACV_2021_paper.pdf)

- **Reinforcement learning based rate adaptation for 360-degree video streaming (2020)** [paper](https://ieeexplore.ieee.org/iel7/11/4359969/09226435.pdf?casa_token=gA-C001tZgwAAAAA:GIv5hK711AKQyDye4ditipKDV9yHPhR4FcxUAxaAqOvUySgDD5yYlI845mOr2GZZh3CgYf3lFBU)

- **RAPT360: Reinforcement learning-based rate adaptation for 360-degree video streaming with adaptive prediction and tiling (2021)** [paper](https://ieeexplore.ieee.org/iel7/76/4358651/09419061.pdf?casa_token=tNMgz3ZO0MoAAAAA:je4Y54RK0jpWCuLktd73lA81LJ3anW5GYckrvJrJTzZUpfqehVhJvO8koRkTeY9HMbi6aOz4HyM)

- **Viewport-Aware Deep Reinforcement Learning Approach for 360&deg Video Caching (2021)** [paper](https://ieeexplore.ieee.org/iel7/6046/4456689/09328310.pdf?casa_token=cvJ_9PDTUDIAAAAA:bn_5ZlQ7ed6jiqUy2kqrBJPEQ1dEoriyaTNaqDcflgG8fObC5xyPeuy8YGSYgSK_FKWtZ7oa874)

- **SAD360: Spherical viewport-aware dynamic tiling for 360-degree video streaming (2022)** [paper](https://ieeexplore.ieee.org/iel7/10008391/10008793/10008862.pdf?casa_token=H5WDeYYp1lwAAAAA:6OzOa-yZrjWwQ31pLKTZ9zl8JLefXnRmCL5fEEgVRSkJhfbb_6Z2fFBsw7clBqFGOijjWx5XU80)

- **Saliency prediction on omnidirectional image with generative adversarial imitation learning (2021)** [paper](https://ieeexplore.ieee.org/iel7/83/9263394/09328187.pdf?casa_token=16U8sZDzHGkAAAAA:AW0tFo39X4EsZDxjijkKjqdpscC_QbD3vNfAseL3Pb5Twan71rruWMzWAw7cpgUHd4-e4RTrldE)

### &hearts; Omnidirectional Vision Tasks

:smiley: ***ODI Generation***

- **Text2Light: Zero-Shot Text-Driven HDR Panorama GenerationText2Light: Zero-Shot Text-Driven HDR Panorama Generation** [Paper](Tasks/Image%26Video%20Manipulation/Omnidirectional%20Image%20Generation%20(Completion)/Text2Light:%20Zero-Shot%20Text-Driven%20HDR%20Panorama%20Generation/Paper.md) [**Code**](https://github.com/FrozenBurning/Text2Light)

- **360-degree image completion by two-stage conditional gans (2019)** [paper](https://ieeexplore.ieee.org/iel7/8791230/8799366/08803435.pdf?casa_token=Du0olyUsxAcAAAAA:w1i89wEo7rz3341FA08-sUmk8Bs-9TH3yVCvxractl8mtI7sSQzsZHnhyg9C3zYPypcQmijB3kY)

- **Spherical image generation from a single image by considering scene symmetry (2021)** [paper](https://ojs.aaai.org/index.php/AAAI/article/download/16242/16049)

- **PanoDiffusion: 360-degree Panorama Outpainting via Diffusion (2024)** [paper](https://openreview.net/forum?id=ZNzDXDFZ0B)

- **Diverse plausible 360-degree image outpainting for efficient 3dcg background creation (2022)** [paper](http://openaccess.thecvf.com/content/CVPR2022/papers/Akimoto_Diverse_Plausible_360-Degree_Image_Outpainting_for_Efficient_3DCG_Background_Creation_CVPR_2022_paper.pdf)

- **Dream360: Diverse and Immersive Outdoor Virtual Scene Creation via Transformer-Based 360&deg Image Outpainting (2024)** [paper](https://ieeexplore.ieee.org/iel7/2945/4359476/10458312.pdf?casa_token=6A-PlnxnPsoAAAAA:LsLx7rvWm6wxCS4hW1yl3gwNeP5rGdwZgw_Ug8LoR_rmWJyP0K46HY-v4PIKj9x6s0p0RqHUsAE)

- **Cylin-Painting: Seamless 360&deg Panoramic Image Outpainting and Beyond (2022)** [paper](https://ieeexplore.ieee.org/iel7/83/4358840/10370742.pdf?casa_token=6N6mKwcMlTwAAAAA:WzYnCR2RdVwEKzWIJHxViSZZwdzwejR4FOssDxA6uyV3ftZI8g_L8x31KML0Mx3jPQ8dsUAuaJo)

- **Guided co-modulated GAN for 360 field of view extrapolation (2022)** [paper](https://ieeexplore.ieee.org/iel7/10044366/10044387/10044439.pdf?casa_token=h_GroX69kB8AAAAA:bfH_JvQImeINkPLfzHH9CoK6mrJ1DT2StS63KVlcIy_dzI7D65ECm6dRiR98Ezo5WHExytL0cX0)

- **PIINET: A 360-degree panoramic image inpainting network using a cube map (2020)** [paper](https://arxiv.org/pdf/2010.16003)

- **Viewport-Oriented Panoramic Image Inpainting (2022)** [paper](https://ieeexplore.ieee.org/iel7/9897158/9897159/09897208.pdf?casa_token=nwDkb9gk8iIAAAAA:BUC1wGOyv-iOz5SVz_LUniXPvhaN5WJ_3bZlNjVNvVEoIr-ulJmte-l_dq8pw6nZxPM4cSfxlOM)

- **Panoramic Image Inpainting with Gated Convolution and Contextual Reconstruction Loss (2024)** [paper](https://ieeexplore.ieee.org/iel7/10445798/10445803/10446469.pdf?casa_token=iUvd4YJUmGEAAAAA:pRft0cCOkI1lRpjTyUX-B8ysYSYHstuQFb9LxG5u2BzZGZ82gqgLrcs4qQzkLjtfg2cwewJaK-w)

- **Text2light: Zero-shot text-driven hdr panorama generation (2022)** [paper](https://dl.acm.org/doi/abs/10.1145/3550454.3555447)

- **Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation (2023)** [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/4522de4178bddb36b49aa26efad537cf-Paper-Conference.pdf)

- **Customizing 360-degree panoramas through text-to-image diffusion models (2024)** [paper](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_Customizing_360-Degree_Panoramas_Through_Text-to-Image_Diffusion_Models_WACV_2024_paper.pdf)

- **Autoregressive Omni-Aware Outpainting for Open-Vocabulary 360-Degree Image Generation (2024)** [paper](https://ojs.aaai.org/index.php/AAAI/article/download/29332/30513)

- **360-degree panorama generation from few unregistered nfov images (2023)** [paper](https://arxiv.org/pdf/2308.14686)

- **Learning to predict indoor illumination from a single image (2017)** [paper](https://arxiv.org/pdf/1704.00090)

- **Deep sky modeling for single image outdoor lighting estimation (2019)** [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hold-Geoffroy_Deep_Sky_Modeling_for_Single_Image_Outdoor_Lighting_Estimation_CVPR_2019_paper.pdf)

- **Neural illumination: Lighting prediction for indoor environments (2019)** [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Song_Neural_Illumination_Lighting_Prediction_for_Indoor_Environments_CVPR_2019_paper.pdf)

- **HDR environment map estimation for real-time augmented reality (2021)** [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Somanath_HDR_Environment_Map_Estimation_for_Real-Time_Augmented_Reality_CVPR_2021_paper.pdf)

- **Stylelight: Hdr panorama generation for lighting estimation and editing (2022)** [paper](https://arxiv.org/pdf/2207.14811)

- **EverLight: Indoor-outdoor editable HDR lighting estimation (2023)** [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Dastjerdi_EverLight_Indoor-Outdoor_Editable_HDR_Lighting_Estimation_ICCV_2023_paper.pdf)

- **Deep outdoor illumination estimation (2017)** [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hold-Geoffroy_Deep_Outdoor_Illumination_CVPR_2017_paper.pdf)

- **Deep parametric indoor lighting estimation (2019)** [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gardner_Deep_Parametric_Indoor_Lighting_Estimation_ICCV_2019_paper.pdf)

- **All-weather deep outdoor lighting estimation (2019)** [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_All-Weather_Deep_Outdoor_Lighting_Estimation_CVPR_2019_paper.pdf)

- **Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image (2020)** [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Inverse_Rendering_for_Complex_Indoor_Scenes_Shape_Spatially-Varying_Lighting_and_CVPR_2020_paper.pdf)

- **Emlight: Lighting estimation via spherical distribution approximation (2021)** [paper](https://ojs.aaai.org/index.php/AAAI/article/download/16440/16247)

- **Gmlight: Lighting estimation via geometric distribution approximation (2022)** [paper](https://ieeexplore.ieee.org/iel7/83/4358840/09725240.pdf?casa_token=tSKIJFCv0isAAAAA:OMiE-4s2rZRqwW5F5ExfgYoCzEYmlxPEwE_WlIvl2eC9CWJkWMJG9chBhqVzCZDHnMoWEjTQcPk)

- **Panoramic image reflection removal (2021)** [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Panoramic_Image_Reflection_Removal_CVPR_2021_paper.pdf)

- **PAR2Net: End-to-End Panoramic Image Reflection Removal (2023)** [paper](https://ieeexplore.ieee.org/iel7/34/4359286/10153662.pdf?casa_token=obu8dPCM29kAAAAA:VZka7wKiQvyPmVRoHUO-h88NpKYp3Ut7FIoD_H6DHKG4Wdb2HPesBnnWF-GA0T-BXqzOe2i4vhg)

- **Zero-shot learning for reflection removal of single 360-degree image (2022)** [paper](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_31)

- **Fully-Automatic Reflection Removal for 360-Degree Images (2024)** [paper](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Fully-Automatic_Reflection_Removal_for_360-Degree_Images_WACV_2024_paper.pdf)

- **360dvd: Controllable panorama video generation with 360-degree video diffusion model (2024)** [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_360DVD_Controllable_Panorama_Video_Generation_with_360-Degree_Video_Diffusion_Model_CVPR_2024_paper.pdf)

- **Conditional 360-degree image synthesis for immersive indoor scene decoration (2023)** [paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Shum_Conditional_360-degree_Image_Synthesis_for_Immersive_Indoor_Scene_Decoration_ICCV_2023_paper.pdf)

:smiley: ***Super-Resolution***

- **Spheresr: 360deg image super-resolution with arbitrary projection via continuous spherical image representation (2022)** [paper](http://openaccess.thecvf.com/content/CVPR2022/papers/Yoon_SphereSR_360deg_Image_Super-Resolution_With_Arbitrary_Projection_via_Continuous_Spherical_CVPR_2022_paper.pdf)

- **Osrt: Omnidirectional image super-resolution with distortion-aware transformer (2023)** [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_OSRT_Omnidirectional_Image_Super-Resolution_With_Distortion-Aware_Transformer_CVPR_2023_paper.pdf)

- **Omnizoomer: Learning to move and zoom in on sphere at high-resolution (2023)** [paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Cao_OmniZoomer_Learning_to_Move_and_Zoom_in_on_Sphere_at_ICCV_2023_paper.pdf)

:smiley: ***Visual Quality Assessment***

- **A framework to evaluate omnidirectional video coding schemes (2015)** [paper](https://ieeexplore.ieee.org/iel7/7327075/7328030/07328056.pdf?casa_token=GdAyZrr2WXUAAAAA:O7gowmGgQoUenNN0QIwu5OnNM_eEIe_t4sqEf9c16DFcxUWaiHKn4ttuDIx5AsrQlekRQAUjfbM)

- **Quality metric for spherical panoramic video (2016)** [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9970/99700C/Quality-metric-for-spherical-panoramic-video/10.1117/12.2235885.short)

- **Weighted-to-spherically-uniform quality evaluation for omnidirectional video (2017)** [paper](https://ieeexplore.ieee.org/iel7/97/7981445/07961186.pdf?casa_token=f8680-qtQdgAAAAA:S1OfbSK34TGdNC2qCowRtXJ_PiTvQ4qs-EyVePrbjRrb3M2wcBFAPXd_73cfZjF27lbRYsl5jNI)

- **Spherical structural similarity index for objective omnidirectional video quality assessment (2018)** [paper](https://ieeexplore.ieee.org/iel7/8472825/8486434/08486584.pdf?casa_token=H-427Fs8ZG0AAAAA:zBBY1h5BWN4Bnoo8e2SdPnHbiaqwMhd8ZYF2R5bSAf-S9Hc-xmYkPRYAO5dm5_l7lQwsVbsUwnM)

- **Weighted-to-spherically-uniform SSIM objective quality evaluation for panoramic video (2018)** [paper](https://ieeexplore.ieee.org/iel7/8648892/8652266/08652269.pdf?casa_token=ULfgMIqeblEAAAAA:bpEOaDzKGSXCBexW7B1GVHAyH8feWsq17uAJmqvcm5MCboS9mIKrdlica7D-XVEWrG7bgXDG3r0)

- **VR IQA NET: Deep virtual reality image quality assessment using adversarial learning (2018)** [paper](https://ieeexplore.ieee.org/iel7/8450881/8461260/08461317.pdf?casa_token=5vQzwoh-eesAAAAA:9dKdQHVxQruK2-ZqL9xOXMpWqKpselvXZ6TWGEkOG9jF4lyfp5vP9xa7cW6zSt0mKSkU_wFg9k4)

- **Deep virtual reality image quality assessment with human perception guider for omnidirectional image (2019)** [paper](https://ieeexplore.ieee.org/iel7/76/4358651/08638985.pdf?casa_token=1nQuMOSjOsoAAAAA:FY8RzQXC9pSnYwN21sBltYLhgSscznWhIwqbwHa_5jtqProKMksre2vYGZKQ0E2-0noGOC0zUVc)

- **TVFormer: Trajectory-guided visual quality assessment on 360° images with transformers (2022)** [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547748)

- **Omnidirectional image quality assessment by distortion discrimination assisted multi-stream network (2021)** [paper](https://ieeexplore.ieee.org/iel7/76/4358651/09432940.pdf?casa_token=nf8MGRNnzw8AAAAA:SmyiS0KUHEJZpFG-3ZoG_9grv1-mR1hvQ76WhVee7IBH7RxPQ_jhjiYLWfxyo4aCBpx3WSr4_fI)

- **MC360IQA: A multi-channel CNN for blind 360-degree image quality assessment (2019)** [paper](https://ieeexplore.ieee.org/iel7/4200690/5418892/08910364.pdf?casa_token=yY_WcPd4z7MAAAAA:xq2UYp-2BU_oRPGlO4cLOFE8mNrDEeOAoORvgnARwfIlkoleWoaJzcOMC_lY33Dh_OlXBWbAIow)

- **Blind omnidirectional image quality assessment with viewport oriented graph convolutional networks (2020)** [paper](https://ieeexplore.ieee.org/iel7/76/4358651/09163077.pdf?casa_token=iKwOGAob7bQAAAAA:3jJv7DRv7H_HYsw34VBtELVeIrpx61BSYYTcs5qiOKh7C3Od4Flqiwg83xPhFhmT1DAmVx1TcA8)

- **Viewport-sphere-Branch Network for Blind Quality Assessment of stitched 360 omnidirectional images (2022)** [paper](https://ieeexplore.ieee.org/iel7/76/4358651/09964240.pdf?casa_token=Dmn66iUvhRgAAAAA:-biPX6_nJYT63JKyuLqVSeklAh4cO8ZDqAJ-3_BnIwhKlmUXizW9x0eKmYRx84UOerodefD1wqQ)

- **Cubemap-based perception-driven blind quality assessment for 360-degree images (2021)** [paper](https://ieeexplore.ieee.org/iel7/83/4358840/09334423.pdf?casa_token=cmerEuk3iCMAAAAA:GjD6SvgUKUslqRhlSeoaA16xHEGIenFnirllMZ8MJEM0E1sLpJigcRQp_pIHPoIdcSiWnBa7oiU)

- **Adaptive hypergraph convolutional network for no-reference 360-degree image quality assessment (2022)** [paper](https://dl.acm.org/doi/pdf/10.1145/3503161.3548337?casa_token=dOEokv2EmVYAAAAA:LU6ctZvadI7kqRVevj5gN468sE8maxdiRSlwwrU6fM6Y8zmWsaSC1ACjI5zYigmHuKH6UzeuaKiI)

- **No-reference quality assessment for 360-degree images by analysis of multifrequency information and local-global naturalness (2021)** [paper](https://ieeexplore.ieee.org/iel7/76/4358651/09432968.pdf?casa_token=wpG9fJYmoQwAAAAA:fDp8-0TBLwk2Z9qJmwTAyVevdEpIX0iJQ9T8cABYn2m0jKTC1Fqmfy0VRi4SKLAJa7UNrkB3gig)

- **Assessor360: Multi-sequence network for blind omnidirectional image quality assessment (2024)** [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ccf4a7323b9ee3e54bf77f0e876b3f8b-Paper-Conference.pdf)

- **Viewport proposal CNN for 360° video quality assessment (2019)** [paper](https://ieeexplore.ieee.org/iel7/8938205/8953184/08953510.pdf?casa_token=0jtqa-UC_hQAAAAA:doY41pZHMY-DzIQ6FY0pDhsfKsPU8N_yyOwDM-N6zXnVDaGIWzHXeL7VDRjMtFjarvvhw1zpRaA)

- **A viewport-driven multi-metric fusion approach for 360-degree video quality assessment (2020)** [paper](https://ieeexplore.ieee.org/iel7/9099125/9102711/09102936.pdf?casa_token=dv-hFT0WvtwAAAAA:ypJwEZ5-83pD3Jrg9F7Vkuu3gsKY-76oCx-Z2LshV4t1ZDK_kVmdSLRUfjQtQBl6sh0BzYZLwbw)

- **Blind quality assessment of omnidirectional videos using spatio-temporal convolutional neural networks (2021)** [paper](https://www.sciencedirect.com/science/article/pii/S0030402620317046)

- **Omnidirectional Video Quality Assessment With Causal Intervention (2024)** [paper](https://ieeexplore.ieee.org/iel7/11/10460203/10380455.pdf?casa_token=sL6TFWJZyocAAAAA:QwT0RxXCy1RbSzAsT969aSc-Xgo8Dkcb4h0WSNxlYCT2xBz-_7zYjujk9C3lx5K951G0Rp3g8GI)

- **Quality assessment for omnidirectional video: A spatio-temporal distortion modeling approach (2020)** [paper](https://ieeexplore.ieee.org/iel7/6046/4456689/09296376.pdf?casa_token=0VW9oEeJLzIAAAAA:s8C-DXNVprZDfo7BCJlKaFhp94DU5U_6-_ycOUB21GiRr6vp87rOBUZ68X1GJLneMmlHPiNjH9Y)

:smiley: ***Object Detection***

- **Object detection for panoramic images based on MS‐RPN structure in traffic road scenes (2019)** [paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/iet-cvi.2018.5304)

- **Object detection in curved space for 360-degree camera (2019)** [paper](https://ieeexplore.ieee.org/iel7/8671773/8682151/08683093.pdf?casa_token=IJYYpE7Ec74AAAAA:eNeAuUkr2AbOyHSvWdAC6s_J0T2vE8yT8UkFeEQGFAvb26e_Fhql82vWeJC-LHsuqiXWn0zvd3o)

- **Training real-time panoramic object detectors with virtual dataset (2021)** [paper](https://ieeexplore.ieee.org/iel7/9413349/9413350/09414503.pdf?casa_token=RgkBTlDz6_4AAAAA:W9qEAw8fMiCtpHPSU2M7WjbxnPfJk2adBcJSBR6tBZwmSFWU2TXwFlrHCbtalewZPK9cI-4OGxY)

- **Learning spherical convolution for fast features from 360 imagery (2017)** [paper](https://proceedings.neurips.cc/paper/2017/file/0c74b7f78409a4022a2c4c5a5ca3ee19-Paper.pdf)

- **Object detection in equirectangular panorama (2018)** [paper](https://ieeexplore.ieee.org/iel7/8527858/8545020/08546070.pdf?casa_token=JkXKvB2Az44AAAAA:lNZDSReWed8imYouDZWmtQvXSgEB5b2MDuSRuG22mxtEqPzS13S1h2GVY-ix8bgA3DlAiO-NtZI)

- **Spherical criteria for fast and accurate 360 object detection (2020)** [paper](https://ojs.aaai.org/index.php/AAAI/article/view/6995/6849)

- **Unbiased iou for spherical image object detection (2022)** [paper](https://ojs.aaai.org/index.php/AAAI/article/download/19929/19688)

- **Field-of-view IoU for object detection in 360&deg images (2023)** [paper](https://ieeexplore.ieee.org/iel7/83/4358840/10190308.pdf?casa_token=GDY6MEvqKlwAAAAA:K7V9xnAq43nOyJiO5Q19zGzIleDO3KvPw9sREbmElINsV_bgnjnFZH1r7iGWCoGs694bjXiN9Zo)

- **Sph2Pob: Boosting Object Detection on Spherical Images with Planar Oriented Boxes Methods. (2023)** [paper](https://www.ijcai.org/proceedings/2023/0137.pdf)

- **Gaussian label distribution learning for spherical image object detection (2023)** [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Gaussian_Label_Distribution_Learning_for_Spherical_Image_Object_Detection_CVPR_2023_paper.pdf)

:smiley: ***Segmentation***

- **Can we pass beyond the field of view? panoramic annular semantic segmentation for real-world surrounding perception (2019)** [paper](https://ieeexplore.ieee.org/iel7/8792328/8813768/08814042.pdf?casa_token=d49sGIdeGhgAAAAA:Y3YMgKKgJDG2YGJA-iX908KCsMFTzRHr5rOvpJtsDKw60Dmgzsa0Bn3oGRpH4C7A1nfvxRpB0ks)

- **Pass: Panoramic annular semantic segmentation (2019)** [paper](https://ieeexplore.ieee.org/iel7/6979/9211608/08835049.pdf?casa_token=A2QJkU5vPwwAAAAA:LyH3MJXGz15J_XGmn2KloMegzfglxTYsFWcJ8hoAh41MkJLM-SpvbMVd6oZW7T2gkiqHBuLK1aA)

- **Ds-pass: Detail-sensitive panoramic annular semantic segmentation through swaftnet for surrounding sensing (2020)** [paper](https://ieeexplore.ieee.org/iel7/9304518/9304528/09304706.pdf?casa_token=oZuBJt38ltIAAAAA:rbRnSian9CC-qiqpJFzmo5cGCNFSbRw1mi-jilDqtfeS04g7WZ2XOJNpOj8D6zl45gHz5-T3bhs)

- **Omnisupervised omnidirectional semantic segmentation (2020)** [paper](https://ieeexplore.ieee.org/iel7/6979/4358928/09204767.pdf?casa_token=aRDyRXLQwbwAAAAA:r77V4JqjYtduGRkn4J5kRKLnPxWca7Gp60vMYgyexjVe4rb-dcrqCJ44-9047q3nTo_jNdDf-8s)

- **Is context-aware CNN ready for the surroundings? Panoramic semantic segmentation in the wild (2021)** [paper](https://ieeexplore.ieee.org/iel7/83/4358840/09321183.pdf?casa_token=YVbmGRDszTQAAAAA:H5uqcr1jsM1Bfbq6XiC2OfJH5dytOlYQ1tqZmqrcBg1gPuCDRHDBUJORoqQbJaDfp9iKSe7tZf8)

- **Capturing omni-range context for omnidirectional segmentation (2021)** [paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Capturing_Omni-Range_Context_for_Omnidirectional_Segmentation_CVPR_2021_paper.pdf)

- **Densepass: Dense panoramic semantic segmentation via unsupervised domain adaptation with attention-augmented context exchange (2021)** [paper](https://arxiv.org/pdf/2108.06383)

- **Transfer beyond the field of view: Dense panoramic semantic segmentation via unsupervised domain adaptation (2021)** [paper](https://ieeexplore.ieee.org/iel7/6979/9826234/09599375.pdf?casa_token=-nyajvdUXxEAAAAA:WtYYWZNwcUEzvfNW4gwfvm03wj_CD_JPu-wta9TGwSP0ygH_YGZjKBROxI6r_uAyfBmd3mu_XAU)

- **Bending reality: Distortion-aware transformers for adapting to panoramic semantic segmentation (2022)** [paper](http://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Bending_Reality_Distortion-Aware_Transformers_for_Adapting_to_Panoramic_Semantic_Segmentation_CVPR_2022_paper.pdf)

- **Behind every domain there is a shift: Adapting distortion-aware vision transformers for panoramic semantic segmentation (2024)** [paper](https://ieeexplore.ieee.org/iel8/34/4359286/10546335.pdf?casa_token=MomRaj-R1x8AAAAA:4jikSblQJ-qpDYdFOesnVU0o7kblUum1DjSHxY2E8NTSE5jqS0fBT_cLOmZnjio0IC-TxN11l8Y)

- **Prototypical pseudo label denoising and target structure learning for domain adaptive semantic segmentation (2021)** [paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Prototypical_Pseudo_Label_Denoising_and_Target_Structure_Learning_for_Domain_CVPR_2021_paper.pdf)

- **Both style and distortion matter: Dual-path unsupervised domain adaptation for panoramic semantic segmentation (2023)** [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_Both_Style_and_Distortion_Matter_Dual-Path_Unsupervised_Domain_Adaptation_for_CVPR_2023_paper.pdf)

- **Look at the neighbor: Distortion-aware unsupervised domain adaptation for panoramic semantic segmentation (2023)** [paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Look_at_the_Neighbor_Distortion-aware_Unsupervised_Domain_Adaptation_for_Panoramic_ICCV_2023_paper.pdf)

- **Semantics Distortion and Style Matter: Towards Source-free UDA for Panoramic Segmentation (2024)** [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_Semantics_Distortion_and_Style_Matter_Towards_Source-free_UDA_for_Panoramic_CVPR_2024_paper.pdf)

- **GoodSAM: Bridging Domain and Capacity Gaps via Segment Anything Model for Distortion-aware Panoramic Semantic Segmentation (2024)** [paper](https://arxiv.org/pdf/2403.16370)

- **SGAT4PASS: spherical geometry-aware transformer for panoramic semantic segmentation (2023)** [paper](https://arxiv.org/pdf/2306.03403)

- **Waymo open dataset: Panoramic video panoptic segmentation (2022)** [paper](https://arxiv.org/pdf/2206.07704)

- **Panoramic panoptic segmentation: Towards complete surrounding understanding via unsupervised contrastive learning (2021)** [paper](https://ieeexplore.ieee.org/iel7/9575127/9575130/09575904.pdf?casa_token=xdY_VJi9q1AAAAAA:DKRFBNJlogbasMNBGL2h7DBRTApglSSV81mP4KS3wYPmAOii6R4Z3V3bwTB5N2ApFamynFkBHLk)

- **Panoramic panoptic segmentation: Insights into surrounding parsing for mobile agents via unsupervised contrastive learning (2023)** [paper](https://ieeexplore.ieee.org/iel7/6979/4358928/10012449.pdf?casa_token=dOQ4dABD6NgAAAAA:GCs0h-PHfp6xpydRcRVtmZK8G4a46quWBiLauqZVVc04-M3OLDl9tebu3CjPsl7wmQag5U98fQs)

- **Panoptic Segmentation from Stitched Panoramic View for Automated Driving (2024)** [paper](https://ieeexplore.ieee.org/iel8/10587320/10588370/10588453.pdf?casa_token=iWZWwHXj-BEAAAAA:2BM-0unREnsKPN6Q2qqX327PHULkebVP6coSBmfi4oKWDlReRwIy1jEM6LCnmIMRxFgH4RdOSNs)

<!-- :smiley: ***Saliency Prediction***

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]() -->

:smiley: ***Depth Estimation***

- **Distortion-Aware Convolutional Filters for Dense Prediction in Panoramic Images (2018)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/DAC/Paper.md) [**Code**](https://github.com/tdsuper/Distortion-aware-CNNs)

- **OmniDepth: Dense Depth Estimation for Indoors Spherical Panoramas (2018)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/OmniDepth/Paper.md) [**Code**](https://github.com/meder411/OmniDepth-PyTorch)
 
- **Spherical View Synthesis for Self-Supervised 360-degree Depth Estimation (2019)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/SphericalVS/Paper.md) [**Code**](https://github.com/VCL3D/SphericalViewSynthesis)

- **Pano Popups: Indoor 3D Reconstruction with a Plane-Aware Network. (2019)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/PanoPopups/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/PanoPopups/Code.md)

- **BiFuse: Monocular 360-degree Depth Estimation via Bi-Projection Fusion (2020)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/BiFuse/Paper.md) [**Code**](https://github.com/Yeh-yu-hsuan/BiFuse)

- **Geometric Structure Based and Regularized Depth Estimation From 360-degree Indoor Imagery (2020)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/GSOMDE/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/GSOMDE/Code.md)

- **PADENet: An Efficient and Robust Panoramic Monocular Depth Estimation Network for Outdoor Scenes (2020)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/PADENet/Paper.md) [Code](https://github.com/zzzkkkyyy/PADENet))

- **UniFuse: Unidirectional Fusion for 360° Panorama Depth Estimation (2021)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/UniFuse/Paper.md) [**Code**](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)
 
- **Improving 360 Monocular Depth Estimation via Non-local Dense Prediction Transformer and Joint Supervised and Self-supervised Learning (2021)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/IMDE/Paper.md) [**Code**](https://github.com/yuniw18/Joint_360depth)

- **SliceNet: deep dense depth estimation from a single indoor panorama using a slice-based representation (2021)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/SliceNet/Paper.md) [**Code**](https://github.com/crs4/SliceNet)

- **PanoDepth: A Two-Stage Approach for Monocular Omnidirectional Depth Estimation (2021)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/PanoDepth/Paper.md) [**Code**](https://github.com/yuyanli0831/PanoDepth)

- **Depth360: Self-supervised Learning for Monocular Depth Estimation using Learnable Camera Distortion Model (2021)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/Depth360/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/Depth360/Code.md)

- **OmniFusion : 360 Monocular Depth Estimation via Geometry-Aware Fusion (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/OmniFusion/Paper.md) [**Code**](https://github.com/yuyanli0831/OmniFusion)
  
- **360MonoDepth: High-Resolution 360° Monocular Depth Estimation (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/360MonoDepth/Paper.md) [**Code**](https://github.com/manurare/360monodepth)

- **ACDNet: Adaptively Combined Dilated Convolution for Monocular Panorama Depth Estimation (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/ACDNet/Paper.md) [**Code**](https://github.com/zcq15/ACDNet)
 
- **GLPanoDepth: Global-to-Local Panoramic Depth Estimation (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/GLPanoDepth/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/GLPanoDepth/Code.md)
 
- **360 Depth Estimation in the Wild: The Depth360 Dataset and the SegFuse Network (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/SegFuse/Paper.md) [**Code**](https://github.com/HAL-lucination/segfuse)
 
- **Deep Depth Estimation on 360° Images with a Double Quaternion Loss (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/DQL/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/DQL/Code.md)

- **PanoFormer: Panorama Transformer for Indoor 360 Depth Estimation (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/PanoPopups/Paper.md) [**Code**](https://github.com/zhijieshen-bjtu/PanoFormer)

- **Multi-Modal Masked Pre-Training for Monocular Panoramic Depth Completion (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/M3PT/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/M3PT/Code.md)

- **Self-supervised indoor 360- degree depth estimation via structural regularization (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/SSIDE/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/SSIDE/Code.md)

- **Variational Depth Estimation on Hypersphere for Panorama (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/OmniVAE/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/OmniVAE/Code.md)

- **SphereDepth: Panorama Depth Estimation from Spherical Domain (2022)**  [Paper](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/SphereDepth/Paper.md) [Code](Tasks/Scene%20Understanding/Monocular%20Depth%20Estimation/SphereDepth/Code.md)

<!-- :smiley: ***Optical Flow Estimation***

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]()

- ** (20)** [paper]() -->

<!-- :smiley: ***Room Layout Estimation and reconstruction*** -->

:smiley: ***Simultaneous Localization And Mapping***

- **360vo: Visual odometry using a single 360 camera (2022)** [paper](https://ieeexplore.ieee.org/iel7/9811522/9811357/09812203.pdf?casa_token=PObyF3j6ZgwAAAAA:iSreJVES5IzEscPNanzsjIF4ZK5zjISGQCAorYq0Rsy0IJMHzXbEwwdMMGWI1m0WFNhZV7xlp0k)

- **Calibrating Panoramic Depth Estimation for Practical Localization and Mapping (2023)** [paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Calibrating_Panoramic_Depth_Estimation_for_Practical_Localization_and_Mapping_ICCV_2023_paper.pdf)

- **360Loc: A Dataset and Benchmark for Omnidirectional Visual Localization with Cross-device Queries (2024)** [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_360Loc_A_Dataset_and_Benchmark_for_Omnidirectional_Visual_Localization_with_CVPR_2024_paper.pdf)

## Citation
If you found our survey helpful for your research, please cite our paper as:

```
@article{Ai2022DeepLF,
  title={Deep Learning for Omnidirectional Vision: A Survey and New Perspectives},
  author={Hao Ai and Zidong Cao and Jin Zhu and Haotian Bai and Yucheng Chen and Ling Wang},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.10468}
}
```

