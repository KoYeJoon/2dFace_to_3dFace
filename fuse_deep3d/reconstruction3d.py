import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import glob
import platform
import argparse
from scipy.io import loadmat, savemat
import time

import sys
import os
sys.path.append("src_3d/")

from src_3d.find_landmark import preprocessing_with_mtcnn
from src_3d.preprocess_img import align_img
from src_3d.utils import *
from src_3d.face_decoder import Face3D
from src_3d.options import Option
import PIL

from SR.src.options.test_options import TestOptions


is_windows = platform.system() == "Windows"



def restore_weights(sess, opt):
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()

    # add batch normalization params into trainable variables
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    # create saver to save and restore weights
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(sess, opt.pretrain_weights)


def demo(args):
    current = time.time()
    # input and output folder
    preprocessing_with_mtcnn()
    image_path = './fuse_deep3d/lm_processed_data'
    save_path = args.objface_results_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    img_list = glob.glob(image_path + '/' + '*.png')
    img_list += glob.glob(image_path + '/' + '*.jpg')
    # read BFM face model
    # transfer original BFM model to our model
    if not os.path.isfile('./BFM/BFM_model_front.mat'):
        transferBFM09()

    # read standard landmarks for preprocessing images
    lm3D = load_lm3d()
    n = 0

    # build reconstruction model
    with tf.Graph().as_default() as graph:

        with tf.device('/cpu:0'):
            opt = Option(is_train=False)
        opt.batch_size = 1
        opt.pretrain_weights = "./BFM/"
        FaceReconstructor = Face3D()
        images = tf.placeholder(name='input_imgs', shape=[opt.batch_size, 224, 224, 3], dtype=tf.float32)
        args.use_pb = 1
        if args.use_pb and os.path.isfile('network/FaceReconModel.pb'):
            print('Using pre-trained .pb file.')
            graph_def = load_graph('network/FaceReconModel.pb')
            tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})
            # output coefficients of R-Net (dim = 257)
            coeff = graph.get_tensor_by_name('resnet/coeff:0')
        else:
            print('Using pre-trained .ckpt file: %s' % opt.pretrain_weights)
            from src_3d import networks
            coeff = networks.R_Net(images, is_training=False)

        # reconstructing faces
        FaceReconstructor.Reconstruction_Block(coeff, opt)
        face_shape = FaceReconstructor.face_shape_t
        face_texture = FaceReconstructor.face_texture
        face_color = FaceReconstructor.face_color
        landmarks_2d = FaceReconstructor.landmark_p
        recon_img = FaceReconstructor.render_imgs
        tri = FaceReconstructor.facemodel.face_buf

        with tf.Session() as sess:
            if not args.use_pb:
                restore_weights(sess, opt)

            print('reconstructing...')
            for file in img_list:
                sess.run(tf.global_variables_initializer())
                n += 1
                print(n)
                # load images and corresponding 5 facial landmarks
                img, lm = load_img(file, file.replace('png', 'txt').replace('jpg', 'txt'))
                # preprocess input image
                input_img, lm_new, transform_params = align_img(img, lm, lm3D)

                coeff_, face_shape_, face_texture_, face_color_, landmarks_2d_, recon_img_, tri_ = sess.run([coeff, \
                                                                                                             face_shape,
                                                                                                             face_texture,
                                                                                                             face_color,
                                                                                                             landmarks_2d,
                                                                                                             recon_img,
                                                                                                             tri],
                                                                                                            feed_dict={
                                                                                                                images: input_img})

                # reshape outputs
                input_img = np.squeeze(input_img)
                face_shape_ = np.squeeze(face_shape_, (0))
                face_texture_ = np.squeeze(face_texture_, (0))
                face_color_ = np.squeeze(face_color_, (0))
                landmarks_2d_ = np.squeeze(landmarks_2d_, (0))
                if not is_windows:
                    recon_img_ = np.squeeze(recon_img_, (0))

                # save output files
                if not is_windows:
                    savemat(os.path.join(save_path,
                                         file.split(os.path.sep)[-1].replace('.png', '.mat').replace('jpg', 'mat')),
                            {'cropped_img': input_img[:, :, ::-1], 'recon_img': recon_img_, 'coeff': coeff_, \
                             'face_shape': face_shape_, 'face_texture': face_texture_, 'face_color': face_color_,
                             'lm_68p': landmarks_2d_, 'lm_5p': lm_new})
                save_obj(os.path.join(save_path,
                                      file.split(os.path.sep)[-1].replace('.png', '_mesh.obj').replace('.jpg',
                                                                                                       '_mesh.obj')),
                         face_shape_, tri_,
                         np.clip(face_color_, 0, 255) / 255)  # 3D reconstruction face (in canonical view)

# if __name__ == '__main__':
#     demo(args)
