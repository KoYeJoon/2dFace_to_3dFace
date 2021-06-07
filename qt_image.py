import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import os
import numpy as np
import glob
# import pywavefront

form_class = uic.loadUiType("./project.ui")[0]

img_path = []
obj_path = ''
pause = False

slider_moved=False
jump_to_frame=0
end=False
escape=0
objimg = np.array([])
input_object = "None"


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



    @pyqtSlot(QPixmap)
    def pixmap_update(self, QPixmap):
        self.label_mainscreen.setPixmap(QPixmap)


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
        elif str == 'btn_start':
            self.btn_start.setEnabled(bool)
        else:
            raise Exception('btn invalid')

    def fileFunction(self):
        global img_path
        img_path_buffer = QFileDialog.getOpenFileName(self, None, None, "Image files (*.png *.jpg)")

        if img_path_buffer[0] is not '' and img_path_buffer != img_path:
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

    def img_load(self):
        pixmap = QPixmap("./data_input/frame.jpg")
        pixmap.load("./data_input/frame.jpg")
        self.label_mainscreen.setPixmap(pixmap)
        return

    # modify modify modify !!!!!!!!!!!!!!!!!!!!!!1
    def obj_load(self):
        # .obj 파일 열어주는 함수
        global obj_path
        # scene = pywavefront.Wavefront(obj_path,strict = True, encoding="iso-8859-1")
        pixmap2 = QPixmap(obj_path)
        pixmap2.load(obj_path)
        self.label_objscreen.setPixmap(pixmap2)
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
        # if os.path.isfile("./data_input/frame.jpg"):
        #     os.remove("./data_input/frame.jpg")
        self.btn_upload.setEnabled(False)
        self.btn_file_reset.setEnabled(True)
        self.btn_start.setEnabled(True)

    def startFunction(self):
        os.system('python main.py')
        self.img_load()
        self.btn_start.setEnabled(False)
        self.btn_confirm.setEnabled(True)

    def confirmFunction(self):
        global input_object
        global obj_path
        input_object = self.spinBox_object_num.value()

        img_list = glob.glob('./fuse_deep3d/data/input/' + '*.png')
        img_list += glob.glob('./fuse_deep3d/data/input/' + '*.jpg')
        total_img = len(img_list)
        print(input_object)

        if 0<input_object and input_object <= total_img :
            # 해당 번호에 해당하는 .obj를 화면에 보여줌 , object reset button 활성화, confirm 버튼 비활성화
            obj_path = './data_output/%06d_mesh.obj'%input_object
            self.obj_load()
            self.btn_object_reset.setEnabled(True)
            self.btn_confirm.setEnabled(False)
        else :
            # 경고창으로 범위 맞게 쓰라고 띄우기
            pass


    def object_resetFunction(self):
        # .obj 파일 화면에서 지워줌 , confirm 버튼 활성화
        pass

    def file_resetFunction(self):
        print("I'm reset")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Annotation_tool = MainWindow()
    Annotation_tool.show()
    sys.exit(app.exec_())
