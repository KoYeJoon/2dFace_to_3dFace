'''
This script enhance all faces in one image with PSFR-GAN and paste it back to the original place.
'''
import dlib
import os
import cv2
import numpy as np 
from tqdm import tqdm
from skimage import transform as trans
from skimage import io

import torch
from utils import utils
from options.test_options import TestOptions
from models import create_model
import glob

def detect_and_align_faces(img, face_detector, lmk_predictor, template_path, template_scale=2, size_threshold=999):
    align_out_size = (512, 512)
    ref_points = np.load(template_path) / template_scale
        
    # Detect landmark points
    face_dets = face_detector(img, 1)

    assert len(face_dets) > 0, 'No faces detected'

    face_dets_arr =[]
    for i, d in enumerate(face_dets):
        face_dets_arr.append([d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()])
    draw_rectangle_dlib(face_dets_arr)

    aligned_faces = []
    tform_params = []
    for det in face_dets:
        if isinstance(face_detector, dlib.cnn_face_detection_model_v1):
            rec = det.rect # for cnn detector
        else:
            rec = det
        if rec.width() > size_threshold or rec.height() > size_threshold: 
            print('Face is too large')
            break
        landmark_points = lmk_predictor(img, rec) 
        single_points = []
        for i in range(5):
            single_points.append([landmark_points.part(i).x, landmark_points.part(i).y])
        single_points = np.array(single_points)
        tform = trans.SimilarityTransform()
        tform.estimate(single_points, ref_points)
        tmp_face = trans.warp(img, tform.inverse, output_shape=align_out_size, order=3)
        aligned_faces.append(tmp_face*255)
        tform_params.append(tform)

    return [aligned_faces, tform_params]


def def_models(opt):
    model = create_model(opt)
    model.load_pretrain_model()
    model.netG.to(opt.device)
    return model


def enhance_faces(LQ_faces, model):
    hq_faces = []
    lq_parse_maps = []
    for lq_face in tqdm(LQ_faces):
        with torch.no_grad():
            lq_tensor = torch.tensor(lq_face.transpose(2, 0, 1)) / 255. * 2 - 1
            lq_tensor = lq_tensor.unsqueeze(0).float().to(model.device)
            output_SR = model.netG(lq_tensor)
        hq_faces.append(utils.tensor_to_img(output_SR, normal=True))
    return hq_faces


def past_faces_back(img, hq_faces, tform_params, upscale=1):
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*upscale), int(h*upscale)), interpolation=cv2.INTER_CUBIC)
    count = 1
    for hq_img, tform in tqdm(zip(hq_faces, tform_params), total=len(hq_faces)):
        tform.params[0:2,0:2] /= upscale
        back_img = trans.warp(hq_img/255., tform, output_shape=[int(h*upscale), int(w*upscale)], order=3) * 255
        
        # blur mask to avoid border artifacts
        mask = (back_img == 0) 
        mask = cv2.blur(mask.astype(np.float32), (5,5))
        mask = (mask > 0)
        img = img * mask + (1 - mask) * back_img
        count += 1
    return img.astype(np.uint8)

count = 1
def save_imgs(img_list, save_dir):
    global count
    for idx, img in enumerate(img_list):
        save_path = os.path.join(save_dir, '{0:06d}.jpg'.format(count))
        io.imsave(save_path, img.astype(np.uint8))
        count += 1


def draw_rectangle_dlib(rects):
    img = cv2.imread('./data_input/frame.jpg')
    count = 1
    for rect in rects :
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        img = cv2.rectangle(img, (x,y),(w,h),(255,0,0),3)
        cv2.putText(img, str(count),(w,h), cv2.FONT_HERSHEY_COMPLEX,1.5, (0,255,255), 1)
        count += 1
    cv2.imwrite('./data_input/frame.jpg', img)

def sr_demo(opt):
    opt.pretrain_model_path = "./SR_pretrain_models/SPARNetHD_V4_Attn2D_net_H-epoch10.pth"

    # opt = TestOptions().parse()
    #  face_detector = dlib.get_frontal_face_detector()
    face_detector = dlib.cnn_face_detection_model_v1('./SR_pretrain_models/mmod_human_face_detector.dat')
    lmk_predictor = dlib.shape_predictor('./SR_pretrain_models/shape_predictor_5_face_landmarks.dat')
    template_path = './SR_pretrain_models/FFHQ_template.npy'

    print('======> Loading images, crop and align faces.')
    img_path = opt.test_img_path

    if opt.type == "image" :
        img_list = [img_path]
    else :
        img_list = glob.glob(img_path + '/' + '*.png')
        img_list += glob.glob(img_path + '/' + '*.jpg')

    save_hq_dir = "./fuse_deep3d/data/input"
    os.makedirs(save_hq_dir, exist_ok=True)

    print('======> Save the enhanced faces.')
    for im in img_list :
        img = dlib.load_rgb_image(im)
        aligned_faces, tform_params = detect_and_align_faces(img, face_detector, lmk_predictor, template_path)
    # Save aligned LQ faces
    # save_lq_dir = os.path.join(opt.results_dir, 'LQ_faces')
    # os.makedirs(save_lq_dir, exist_ok=True)
    # print('======> Saving aligned LQ faces to', save_lq_dir)
    # save_imgs(aligned_faces, save_lq_dir)

        enhance_model = def_models(opt)
        hq_faces = enhance_faces(aligned_faces, enhance_model)
    # Save LQ parsing maps and enhanced faces
    # save_hq_dir = os.path.join(opt.results_dir, 'HQ')

        save_imgs(hq_faces, save_hq_dir)
    # print("hq_faces***************************************************************")
    # print(hq_faces)
    # print('======> Paste the enhanced faces back to the original image.')
    # hq_img = past_faces_back(img, hq_faces, tform_params, upscale=opt.test_upscale)
    # final_save_path = os.path.join(opt.results_dir, 'hq_final.jpg')
    # print('======> Save final result to', final_save_path)
    # io.imsave(final_save_path, hq_img)
    return opt

if __name__ == '__main__':
    opt = TestOptions().parse()
    sr_demo(opt)
