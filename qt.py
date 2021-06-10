# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar)
import cv2
import os
import numpy as np
import glob
from time import sleep
import threading
import shutil
import dlib


form_class = uic.loadUiType("./project.ui")[0]
obj_list = []
img_path = []
obj_path = ''
video_path = ''
pause = False

slider_moved=False
jump_to_frame=0
end=False
escape=0
objimg = np.array([])
input_object = "None"

framecount = 0
slider_moved = False
jumped = False
token = 0

flush = False

class SignalOfTrack(QObject):
    frameCount = pyqtSignal(int)
    buttonName = pyqtSignal(str, bool)
    pixmapImage = pyqtSignal(QPixmap)

    def slider_run(self, int):
        self.frameCount.emit(int)

    def btn_run(self, str, bool):
        self.buttonName.emit(str, bool)

    def pixmap_run(self, QPixmap):
        self.pixmapImage.emit(QPixmap)

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # about file
        self.btn_file.clicked.connect(self.fileFunction)
        self.btn_file_reset.clicked.connect(self.file_resetFunction)
        self.btn_upload.clicked.connect(self.uploadFunction)

        # detect, reconstruct
        self.btn_start.clicked.connect(self.startFunction)
        self.btn_confirm.clicked.connect(self.confirmFunction)
        self.btn_object_reset.clicked.connect(self.object_resetFunction)

        # video
        self.btn_play.clicked.connect(self.play)
        self.btn_video_file.clicked.connect(self.videofileFunction)
        self.btn_upload_2.clicked.connect(self.uploadVideoFunction)

        self.horizontalSlider.sliderMoved.connect(self.slider_moved)
        self.horizontalSlider.sliderReleased.connect(self.slider_released)
        self.horizontalSlider.sliderPressed.connect(self.slider_pressed)

        self.btn_video_start.clicked.connect(self.my_thread)

        self.btn_video_capture.clicked.connect(self.save_capture)

        self.spinBox_object_num.setValue(1)

    @pyqtSlot(QPixmap)
    def pixmap_update(self, QPixmap):
        self.label_mainscreen.setPixmap(QPixmap)

    @pyqtSlot(int)
    def slider_control(self, int):
        self.horizontalSlider.setValue(int)

    @pyqtSlot(str, bool)
    def btn_control(self, str, bool):
        if str == 'btn_file':
            self.btn_file.setEnabled(bool)
        elif str == 'btn_file_reset':
            self.btn_file_reset.setEnabled(bool)
        elif str == 'btn_upload':
            self.btn_upload.setEnabled(bool)
        elif str == 'btn_confirm':
            self.btn_confirm.setEnabled(bool)
        elif str == 'btn_object_reset':
            self.btn_object_reset.setEnabled(bool)
        elif str == 'btn_startn_video_stt':
            self.btn_start.setEnabled(bool)
        elif str == 'btn_play':
            self.btn_play.setEnabled(bool)
        elif str == 'btn_video_file':
            self.btn_video_file.setEnabled(bool)
        elif str == 'btn_video_capture':
            self.btn_video_capture.setEnabled(bool)
        else:
            raise Exception('btn invalid')

    def fileFunction(self):
        global img_path
        img_path_buffer = QFileDialog.getOpenFileName(self, None, None, "Image files (*.png *.jpg)")

        if img_path_buffer[0] is not '':
            img_path = img_path_buffer
            self.label_file_path.setText(img_path[0])

        else:
            pass
        if not img_path:
            self.btn_upload.setEnabled(False)
        else:
            if img_path_buffer[0] == '':
                return
            else:
                self.btn_upload.setEnabled(True)
                self.btn_file_reset.setEnabled(True)
                return

    def videofileFunction(self):
        global video_path
        video_path_buffer = QFileDialog.getOpenFileName(self, None, None, "Video files (*.mp4)")
        if video_path_buffer[0] is not '':
            video_path = video_path_buffer
            self.label_file_path_2.setText(video_path[0])
            # self.vid_load()
            temp_vid = cv2.VideoCapture(video_path[0])
            num_of_frame = int(temp_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.horizontalSlider.setMinimum(0)
            self.horizontalSlider.setMaximum(num_of_frame)
            self.label_end_frame.setText('%d' % num_of_frame)
            temp_vid.release()
        else:
            pass
        if not video_path:
            self.btn_upload_2.setEnabled(False)
        else:
            if video_path_buffer[0] == '':
                return
            else:
                self.btn_upload_2.setEnabled(True)
                self.btn_file_reset.setEnabled(True)
                return

    def img_load(self):
        pixmap = QPixmap("./data_input/frame.jpg")
        pixmap.load("./data_input/frame.jpg")
        self.label_mainscreen.setPixmap(pixmap)
        return


    def uploadFunction(self):
        global pause
        global img_path
        pause = False
        img = cv2.imread(img_path[0],cv2.IMREAD_COLOR)
        if not os.path.isdir('./data_input'):
            os.mkdir('./data_input')
        cv2.imwrite("./data_input/frame.jpg", img)

        self.img_load()
        self.btn_video_file.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_file_reset.setEnabled(True)
        self.btn_start.setEnabled(True)

    def uploadVideoFunction(self):
        global flush
        global pause
        flush = False
        pause = False
        # self.vid_load()
        print(video_path[0])
        global cap
        cap = cv2.VideoCapture(video_path[0])  # 저장된 영상 가져오기 프레임별로 계속 가져오는 듯
        k = cap.isOpened()
        if k == False:
            cap.open(video_path[0])
        self.ret, self.frame = cap.read()
        self.rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.convertToQtFormat = QImage(self.rgbImage.data, self.rgbImage.shape[1], self.rgbImage.shape[0],
                                        QImage.Format_RGB888)
        pixmap = QPixmap(self.convertToQtFormat)
        self.p = pixmap.scaled(400, 300, QtCore.Qt.IgnoreAspectRatio)
        self.label_mainscreen.setPixmap(pixmap)
        self.btn_play.setEnabled(False)

        cap.release()
        cv2.destroyAllWindows()

        self.btn_video_file.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_upload_2.setEnabled(False)

        self.btn_file_reset.setEnabled(True)
        self.btn_video_start.setEnabled(True)
        return

    def startFunction(self):
        global obj_list

        # fuse_3d/data/input 에 data 있으면 삭제
        if os.path.isdir('./fuse_deep3d/data') :
            shutil.rmtree('./fuse_deep3d/data')
        # fuse_3d/lm_processed_data 에 data 있으면 삭제
        if os.path.isdir('./fuse_deep3d/lm_processed_data'):
            shutil.rmtree('./fuse_deep3d/lm_processed_data')
        if os.path.isdir('./data_output'):
            shutil.rmtree('./data_output')

        img = cv2.imread('./data_input/frame.jpg', cv2.IMREAD_COLOR)
        face_detector = dlib.cnn_face_detection_model_v1('./SR_pretrain_models/mmod_human_face_detector.dat')
        face_dets = face_detector(img, 1)

        if len(face_dets) > 0 :
            os.system('python main.py --pyqt_ver pyqt')
            self.img_load()
            self.btn_start.setEnabled(False)
            self.btn_confirm.setEnabled(True)
            obj_list = glob.glob('./data_output/' + '*.obj')
        else :
            reply = QMessageBox.question(self, "can't detect face", "we can't detect face", QMessageBox.Ok)


    def confirmFunction(self):
        global input_object
        global obj_path
        global obj_list
        input_object = self.spinBox_object_num.value()

        img_list = glob.glob('./fuse_deep3d/data/input/' + '*.png')
        img_list += glob.glob('./fuse_deep3d/data/input/' + '*.jpg')

        obj_path = './data_output/%06d_mesh.obj' % input_object
        if obj_path in obj_list :
            self.btn_object_reset.setEnabled(True)
            self.btn_confirm.setEnabled(False)
            os.system('meshlab '+obj_path)
        elif input_object < 1 and input_object > len(img_list):
            reply = QMessageBox.question(self, 'Out of Boundary', 'out of boundary', QMessageBox.Ok)
        else :
            reply = QMessageBox.question(self, "can't make 3d reconstruction", "It can't make 3d reconstruction, please try another number", QMessageBox.Ok)


    def object_resetFunction(self):
        # confirm 버튼 활성화
        self.btn_object_reset.setEnabled(False)
        self.btn_confirm.setEnabled(True)


    def file_resetFunction(self):
        self.flush()
        self.btn_file.setEnabled(True)
        self.btn_video_file.setEnabled(True)
        self.btn_video_capture.setEnabled(False)
        pixmap = QPixmap()
        self.label_mainscreen.setPixmap(pixmap)
        self.label_file_path.setText('File Path')
        self.label_file_path_2.setText('File Path')
        self.label_end_frame.setText('end')

    def video2Frame(self):
        signal = SignalOfTrack()
        signal.frameCount.connect(self.slider_control)
        signal.buttonName.connect(self.btn_control)
        signal.pixmapImage.connect(self.pixmap_update)
        global cap
        cap = cv2.VideoCapture(video_path[0])
        k = cap.isOpened()
        if k == False:
            cap.open(video_path[0])

        global framecount, pause_flag, slider_moved, pause, jumped, token
        framecount = 0.0
        pause_flag = 0
        jump_count = None
        token = 0
        while True:
            if pause:
                framecount = cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.btn_video_capture.setEnabled(True)
                pass
            else:
                self.btn_video_capture.setEnabled(False)

                ret, img = cap.read()
                if ret:

                    self.rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.convertToQtFormat = QImage(self.rgbImage.data, self.rgbImage.shape[1], self.rgbImage.shape[0],
                                                    QImage.Format_RGB888)

                    self.pixmap = QPixmap(self.convertToQtFormat)
                    self.p = self.pixmap.scaled(400, 300, QtCore.Qt.IgnoreAspectRatio)

                    self.label_mainscreen.setPixmap(self.p)

                    self.label_mainscreen.update()

                    sleep(0.03)
                    self.btn_play.setEnabled(True)
                else:
                    break
                if not ret:
                    self.play()
                    self.video_end()
                else:
                    framecount = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if jumped:
                if jump_to_frame > 8:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame - 8)
                    framecount = cap.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    framecount = cap.get(cv2.CAP_PROP_POS_FRAMES)

                jump_count = 0
                jumped = False

            if jump_count is None:
                pass
            elif jump_count <= 8:
                jump_count += 1
                signal.btn_run('btn_play', False)
                pause_flag = 1
                pass
            elif jump_count > 8:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                framecount = cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.play()
                signal.btn_run('btn_play', True)
                jump_count = None
                pause_flag = 0

            while pause:
                self.btn_video_capture.setEnabled(True)

                if slider_moved:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame)
                    ret, img = cap.read()
                    if ret:
                        h, w, ch = img.shape
                        bytesPerLine = ch * w
                        qimg_3 = QImage(img, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                        signal.pixmap_run(QPixmap.fromImage(qimg_3))
                        framecount = jump_to_frame
                        slider_moved = False
                    else:
                        pass
                else:
                    pass

                signal.slider_run(framecount)

                if jumped or pause_flag:
                    break
                if flush:
                    return
                if not pause:
                    token = 1
                    break
            if token:
                token = 0
                continue
            if jumped:
                continue
            # original_image = img
            signal.slider_run(framecount)

            if not ret:
                if not pause:
                    print(ret, pause)
                    self.play()
                self.video_end()
                continue
            else:
                pass

            if jump_count is None:
                signal.btn_run('btn_play', True)
            else:
                pass

        cap.release()
        cv2.destroyAllWindows()

    def my_thread(self):
        self.btn_play.setChecked(True)
        self.btn_video_capture.setEnabled(False)
        self.btn_play.setText('Pause\n(space)')
        self.horizontalSlider.setEnabled(False)
        self.btn_play.setEnabled(True)
        self.btn_video_start.setEnabled(False)

        th = threading.Thread(target=self.video2Frame)
        th.setDaemon(True)
        th.start()

    def play(self):
        global pause
        if not pause:
            self.horizontalSlider.setEnabled(True)
            pause = True
            self.btn_play.setChecked(False)
            self.btn_play.setText('Play\n(space)')
            self.btn_play.setShortcut(Qt.Key.Key_Space)
            self.btn_video_capture.setEnabled(True)
            self.centralwidget.setFocus()

        else:
            self.horizontalSlider.setEnabled(False)
            pause = False
            self.btn_play.setChecked(True)
            self.btn_play.setText('Pause\n(space)')
            self.btn_video_capture.setEnabled(False)
            self.btn_play.setShortcut(Qt.Key.Key_Space)
            # self.btn_tab.setEnabled(True)
        return

    def video_end(self):
        global end
        self.horizontalSlider.setValue(self.horizontalSlider.maximum())
        end = True
        self.btn_file_reset.setEnabled(True)
        return

    def slider_pressed(self):
        signal = SignalOfTrack()
        signal.frameCount.connect(self.slider_control)
        signal.slider_run(self.horizontalSlider.value())
        global framecount
        framecount = self.horizontalSlider.value()
        pass

    def slider_moved(self):
        global slider_moved, jump_to_frame, end
        slider_moved = True
        jump_to_frame = self.horizontalSlider.value()
        end = False

    def slider_released(self):
        global slider_moved, jump_to_frame, escape, objimg
        slider_moved = True
        jump_to_frame = self.horizontalSlider.value()
        escape = 1
        objimg = np.array([])

    def save_capture(self):
        int_framecount = int(framecount)
        print(int_framecount)

        cap = cv2.VideoCapture(video_path[0])
        cap.set(1, int_framecount)
        ret, frame = cap.read()
        cv2.imwrite("./data_input/frame.jpg", frame)

        self.btn_start.setEnabled(True)


    def flush(self):
        global input_object
        global end
        global flush
        global pause

        if pause:
            pass
        else:
            self.play()

        flush = True
        if os.path.isfile("./data_input/frame.jpg"):
            os.remove("./data_input/frame.jpg")
        self.img_load()
        pixmap = QPixmap("./data_input/frame.jpg")
        self.label_mainscreen.setPixmap(pixmap)
        self.btn_file.setEnabled(True)
        self.btn_video_file.setEnabled(True)
        self.btn_upload.setEnabled(False)
        self.btn_upload_2.setEnabled(False)

        self.spinBox_object_num.setValue(1)
        self.btn_confirm.setEnabled(False)
        self.btn_object_reset.setEnabled(False)
        self.btn_file_reset.setEnabled(False)
        self.btn_play.setChecked(False)
        self.btn_play.setEnabled(False)
        self.btn_video_start.setEnabled(False)
        end = False
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(0)
        self.horizontalSlider.setEnabled(False)
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Annotation_tool = MainWindow()
    Annotation_tool.show()
    sys.exit(app.exec_())
