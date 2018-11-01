# -*- coding: UTF-8 -*-
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPushButton
import qtawesome
import math
import cv2
import dlib
import numpy as np
import sys

predictor_path = "resources/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR = 0.5

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))

POINTS = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS
ALIGN_POINTS = POINTS
OVERLAY_POINTS = [POINTS]


def create_capture(source=0):
    source = str(source).strip()
    chunks = source.split(':')

    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = int(chunks[0])
    return cv2.VideoCapture(source)


# 获取摄像头
def get_cam_frame(cam):
    ret, img = cam.read()
    img = cv2.resize(img, (400, 225))  # (800, 450)
    return img


# 提取特征点
def get_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return -1

    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


# 注释标记点
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


# 找到点集的凸包
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)  # 填充凸多边形


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0
    im = im * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


# 归一化
def transformation_f_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)

    points1 /= s1
    points2 /= s2

    u, s, vt = np.linalg.svd(points1.T * points2)
    r = (u * vt).T

    h_stack = np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T))
    return np.vstack([h_stack, np.matrix([0., 0., 1.])])


def get_im_w_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s


# 将图像变形
def warp_im(im, m, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, m[:2], (dshape[1], dshape[0]), dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_im


# 修正颜色
def correct_colours(im1, im2, landmarks1):
    mean_left = np.mean(landmarks1[LEFT_EYE_POINTS], axis=0)
    mean_right = np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)

    blur_amount = COLOUR_CORRECT_BLUR * np.linalg.norm(mean_left - mean_right)
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # avoid division errors
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


# 得到旋转点
def get_rotated_points(point, anchor, deg_angle):
    angle = math.radians(deg_angle)
    px, py = point
    ox, oy = anchor

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [int(qx), int(qy)]


# 融合透明度
def blend_w_transparency(face_img, overlay_image):
    # BGR
    overlay_img = overlay_image[:, :, :3]
    # A
    overlay_mask = overlay_image[:, :, 3:]

    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)  # 将图像从一个颜色空间转换为另一个颜色空间
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # cast to 8 bit matrix
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


# 叠加眼镜
def glasses_filter(face, glasses):
    landmarks = get_landmarks(face)
    pts1 = np.float32([[0, 0], [599, 0], [0, 208], [599, 208]])
    if type(landmarks) is not int:
        """
        GLASSES ANCHOR POINTS:

        17 & 26 edges of left eye and right eye (left and right extrema)
        0 & 16 edges of face across eyes (other left and right extra, interpolate between 0 & 17, 16 & 26 for half way points)
        19 & 24 top of left and right brows (top extreme)
        27 is centre of the eyes on the nose (centre of glasses)
        28 is the bottom threshold of glasses (perhaps interpolate between 27 & 28 if too low) (bottom extreme)
        """

        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        glasses_width = face_width * 1.0

        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height * 1.0

        # generate bounding box from the anchor points
        anchor_point = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, anchor_point, face_angle)

        tr = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, anchor_point, face_angle)

        bl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, anchor_point, face_angle)

        br = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, anchor_point, face_angle)

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)

        rotated = cv2.warpPerspective(glasses, m, (face.shape[1], face.shape[0]))  # 对图像进行透视变换
        result_2 = blend_w_transparency(face, rotated)

        return result_2

    else:
        please = cv2.imread('resources/please1.png', -1)
        please = cv2.resize(please, (640, 480))
        return please


class Ui_MainWindow(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # self.face_recong = face.Recognition()
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.glasses = cv2.imread('resources/glass1.png', -1)
        # self.setWindowOpacity(0.95)  # 设置窗口透明度

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()  # 创建主部件的box布局

        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u' ')
        self.button_open_camera.setStyleSheet("QPushButton{border-image: url(resources/camera.png)}"
                                              "QPushButton:hover{border-image: url(resources/camera.png)}")

        self.button_glass1 = QtWidgets.QPushButton(u' ')
        self.button_glass1.setStyleSheet("QPushButton{border-image: url(resources/glass1.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass1.png)}")
        self.button_glass2 = QtWidgets.QPushButton(u' ')
        self.button_glass2.setStyleSheet("QPushButton{border-image: url(resources/glass2.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass2.png)}")
        self.button_glass3 = QtWidgets.QPushButton(u' ')
        self.button_glass3.setStyleSheet("QPushButton{border-image: url(resources/glass3.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass3.png)}")
        self.button_glass4 = QtWidgets.QPushButton(u' ')
        self.button_glass4.setStyleSheet("QPushButton{border-image: url(resources/glass4.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass4.png)}")
        self.button_glass5 = QtWidgets.QPushButton(u' ')
        self.button_glass5.setStyleSheet("QPushButton{border-image: url(resources/glass5.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass5.png)}")
        self.button_glass6 = QtWidgets.QPushButton(u' ')
        self.button_glass6.setStyleSheet("QPushButton{border-image: url(resources/glass6.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass6.png)}")
        self.button_glass7 = QtWidgets.QPushButton(u' ')
        self.button_glass7.setStyleSheet("QPushButton{border-image: url(resources/glass7.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass7.png)}")
        self.button_glass8 = QtWidgets.QPushButton(u' ')
        self.button_glass8.setStyleSheet("QPushButton{border-image: url(resources/glass8.png)}"
                                         "QPushButton:hover{border-image: url(resources/glass8.png)}")

        self.button_close = QtWidgets.QPushButton(u' ')
        self.button_close.setStyleSheet("QPushButton{border-image: url(resources/exit.png)}"
                                        "QPushButton:hover{border-image: url(resources/exit.png)}")

        self.button_glass1.setMinimumHeight(60)
        self.button_glass2.setMinimumHeight(60)
        self.button_glass3.setMinimumHeight(60)
        self.button_glass4.setMinimumHeight(60)
        self.button_glass5.setMinimumHeight(60)
        self.button_glass6.setMinimumHeight(60)
        self.button_glass7.setMinimumHeight(60)
        self.button_glass8.setMinimumHeight(70)
        self.button_open_camera.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 0)

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_glass1)
        self.__layout_fun_button.addWidget(self.button_glass2)
        self.__layout_fun_button.addWidget(self.button_glass3)
        self.__layout_fun_button.addWidget(self.button_glass4)
        self.__layout_fun_button.addWidget(self.button_glass5)
        self.__layout_fun_button.addWidget(self.button_glass6)
        self.__layout_fun_button.addWidget(self.button_glass7)
        self.__layout_fun_button.addWidget(self.button_glass8)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'崇新学堂')

        self.setWindowTitle('崇新学堂虚拟试戴')
        self.setWindowIcon(QIcon('resources/sdu.png'))

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)

        self.button_glass1.clicked.connect(self.button_glass1_click)
        self.button_glass2.clicked.connect(self.button_glass2_click)
        self.button_glass3.clicked.connect(self.button_glass3_click)
        self.button_glass4.clicked.connect(self.button_glass4_click)
        self.button_glass5.clicked.connect(self.button_glass5_click)
        self.button_glass6.clicked.connect(self.button_glass6_click)
        self.button_glass7.clicked.connect(self.button_glass7_click)
        self.button_glass8.clicked.connect(self.button_glass8_click)

        self.timer_camera.timeout.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)
        self.cxxt = QPushButton(u' ', self)
        self.cxxt.setStyleSheet("QPushButton{border-image: url(resources/cxxt.png)}"
                                "QPushButton:hover{border-image: url(resources/cxxt.png)}")
        self.cxxt.move(380, 0)
        self.cxxt.setMinimumHeight(100)
        self.cxxt.setMinimumWidth(300)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                # if msg==QtGui.QMessageBox.Cancel:
                #   pass
            else:
                self.timer_camera.start(30)

                self.button_open_camera.setText(u' ')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u' ')

    def button_glass1_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass1.png', -1)

    def button_glass2_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass2.png', -1)

    def button_glass3_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass3.png', -1)

    def button_glass4_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass4.png', -1)

    def button_glass5_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass5.png', -1)

    def button_glass6_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass6.png', -1)

    def button_glass7_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass7.png', -1)

    def button_glass8_click(self):
        if self.timer_camera.isActive() == True:
            self.glasses = cv2.imread('resources/glass8.png', -1)

    def show_camera(self):
        cam = self.cap.read()
        flag, self.image = cam
        '''
        landmarks = get_landmarks(self.image)
        if type(landmarks) is not int:
            show = annotate_landmarks(self.image, landmarks)
        '''
        self.image = cv2.resize(self.image, (640, 480))
        result_2 = glasses_filter(self.image, self.glasses)

        show = cv2.cvtColor(result_2, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')

        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setObjectName("MainWindow")
    ui.setStyleSheet("#MainWindow{border-image:url(resources/background2.png);}")
    ui.show()
    sys.exit(app.exec_())
