# 2Dface_to_3Dface

| name | how to contact |
| --- | --- |
| Ko Ye Joon | [![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:yejoon.ko@gmail.com)](mailto:yejoon.ko@gmail.com) |
| Jeong Seung Won |  [![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:jeongsw34@gmail.com)](mailto:jeongsw34@gmail.com) | 


This is the project about super resolution, face reconstruction, face frontalization.  
we merged [sparNet](https://github.com/chaofengc/Face-SPARNet) and [deep3d model](https://github.com/microsoft/Deep3DFaceReconstruction).  
  
If you give image which contained one or more person to model, the model will produce 3d reconstruction face and frontalization face for image.


<br>
<b>(the system structure)</b>
<br>

![model structure](./images/model_structure.png)


<br>


### Reference 
super resolution - sparNet : [click to link](https://github.com/chaofengc/Face-SPARNet)  
3d reconstruction - deep3d : [click to link](https://github.com/microsoft/Deep3DFaceReconstruction)


<br>

### 1. requirements

##### 1-1. Install Dependencies

```
* conda create -n face3d python=3.6
* conda activate face3d
* tensorflow-gpu == 1.12.0 (conda install tensorflow-gpu == 1.12.0 )
* keras==2.2.4
* torch==1.5.1 #pip uninstall tensorboard
* torchvision==0.6.1
* mtcnn
* pillow
* argparse
* scipy
* scikit-image
* imgaug
* opencv-python
* dlib
* tqdm
* PyQt5==5.15.1
```

or
```
* conda env create -f environment.yaml
* conda activate face3d
```

<br>

##### 1-2. compile tf_mesh_renderer

we referred to <b>[this site](https://github.com/microsoft/Deep3DFaceReconstruction) </b>as the link.

```
$ git clone https://github.com/google/tf_mesh_renderer.git
$ cd tf_mesh_renderer
$ git checkout ba27ea1798
$ git checkout master WORKSPACE
```  
  
<br />

set <b>-D_GLIBCXX_USE_CXX11_ABI=1</b> in ./mesh_renderer/kernels/BUILD  
  
```
$ bazel test ...
```
  
  

##### 1-3. project directory structure
```
.
2dFace_to_3dFace
???   README.md
???   main.py    
???   environment.yaml
???
????????????BFM
???      BFM_model_front.mat
???      similarity_Lm3D_all.mat
???   
????????????data_input
???       video/image for input
???
????????????network
???       FaceReconModel.pb
???   
????????????fuse_deep3d
???        data
???        ???     ????????????input # SR results images will be saved here
???        ???   
???        ????????????renderer
???        ???          rasterize_triangles_kernel.so
???        ???          ...
???        ??? 
???        ????????????src_3d
????????????SR_pretrain_models
???           FFHQ_template.npy
???           mmod_human_face_detector.dat
???           shape_predictor_5_face_landmarks.dat
???           SPARNetHD_V4_Attn2D_net_H-epoch10.pth       
????????????SR
???    ????????????src
???          ...
???
...
```


### 2. What do you need to make 3d reconstruction face? 

#### clone the repository
```
$ git clone https://github.com/KoYeJoon/2dFace_to_3dFace.git
$ cd 2dFace_to_3dFace
```

#### for testing you have to ...
  
<br>

##### i) download Basel Face Model.
download "01_MorphableModel.mat" in [this site](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) and put into ./fuse_deep3d/BFM

<br>

##### ii) download CoarseData 
Download Coarse data in the first row of Introduction part in [their repository](https://github.com/Juyong/3DFace) and put "Exp_Pca.bin" into ./fuse_deep3d/BFM. 

<br>

##### iii) put the compiled rasterize_triangles_kernel.so into ./renderer folder.

<br>

##### iv) Download Face-SPARNet pretrained models
Download Face-SPARNet pretrained models in from the following link and put then in *./SR_pretrain_models*

* [GoogleDrive](https://drive.google.com/drive/folders/1PZ_TP77_rs0z56WZausgK0m2oTxZsgB2)
* [BaiduNetDisk](https://pan.baidu.com/share/init?surl=zYimaAnIgMIKBf9KANpxog)

<br>

##### v) if you want to get pre-trained model, download pre-trained reconstruction network. 
download in [this link](https://drive.google.com/file/d/176LCdUDxAj7T2awQ5knPMPawq5Q2RUWM/view) and put "FaceReconModel.pb" into ./network subfolder. 


<br>
<br>


### How to make 2d face to 3d face?
you can use easily !!

1. use terminal
   1-1. put your images in ./data_input directory
```
(Precautions)
* you can change name of input_directory. 
but if you change name, you have to give argument when you run main.py
* if you want to give a input type='image', you can skip. 
but you have to give argument(--type image --test_img_path ./your_img_path) when you run.
```   
 
   1-2. run!!

```
python main.py [--arguments]
```

Below is argument list.
```
[--type] : image/ dir, default : dir
[--test_img_path] : your custom image input image/dir path, default : ./data_input
[--objface_results_dir] : where to save .obj face files, default : ./data_output
```  
  
<br>  


2. use pyqt
   2-1. run !!
```
python qt.py
```


<br>
<br>
<br>


### 3. revision history

* 21.06.04. extend input type(you can try not only image but also image directory)
* 21.06.07. extend gui program (use pyqt), input type(image)
* 21.06.10. extend gui program, input type(image, video)



<br>
<br>
<br>

  
### 4. Things to improve
* Currently, there are many overlapping codes, with the face recognition process. It seems necessary to modify the code for this part to make the program lighter.
* In the case of video, it has the advantage of collecting multiple photos of one person. we can think about whether there is 
a way to use this advantage to make 3d reconstruction more precise.
  
* In the case of deep3d, we can see that 3d reconstruction was not done well for Asians, which needs to be
supplemented by further research.
  



<br>
<br>
<br>


# Citation
```
@inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
}

@InProceedings{ChenSPARNet,
    author = {Chen, Chaofeng and Gong, Dihong and Wang, Hao and Li, Zhifeng and Wong, Kwan-Yee K.},
    title = {Learning Spatial Attention for Face Super-Resolution},
    Journal = {IEEE Transactions on Image Processing (TIP)},
    year = {2020}
}
```
