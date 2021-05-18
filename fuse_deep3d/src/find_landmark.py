import cv2
import argparse
import os
import glob
from mtcnn import MTCNN


def parse_args():
    desc = "Data preprocessing for Deep3DRecon."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img_path', type=str, default='./data/input', help='original images folder')
    parser.add_argument('--save_path', type=str, default='./data/preprocess_data/', help='custom path to save proccessed images and labels')
    parser.add_argument('--opt', type=str, default='test', help='train/test mode')

    return parser.parse_args()


def preprocessing_with_mtcnn(args):
    image_path = args.img_path
    save_path = args.save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    img_list = sorted(glob.glob(image_path + '/' + '*.png'))
    img_list += sorted(glob.glob(image_path + '/' + '*.jpg'))


    file_n = 1
    for file in img_list:
        img = cv2.imread(file)
        #img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        print(file)
        detector = MTCNN()
        faces = detector.detect_faces(img)

        if len(faces) > 0:
            count = 1
            for result in faces :
                features = [result['keypoints']['left_eye'], result['keypoints']['right_eye'],result['keypoints']['nose'], result['keypoints']['mouth_left'],
                            result['keypoints']['mouth_right']]
                if args.opt == 'train' :
                    cv2.imwrite('%s%06d.jpg' % (save_path, file_n), img)
                    with open('%s%06d.txt' % (save_path, file_n), "a") as f:
                        for i in features:
                            print(str(i[0]) + ' ' + str(i[1]), file=f)

                else :
                    cv2.imwrite('%s%06d-%03d.jpg' % (save_path, file_n, count), img)
                    with open('%s%06d-%03d.txt' % (save_path, file_n, count), "a") as f:
                        for i in features:
                            print(str(i[0]) + ' ' + str(i[1]), file=f)
                    count += 1
            file_n += 1



if __name__ == '__main__':
    args = parse_args()
    preprocessing_with_mtcnn(args)
