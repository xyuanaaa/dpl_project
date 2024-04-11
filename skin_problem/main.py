from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QSize
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import cv2
import numpy as np
from ui_file.mainwindow import Ui_MainWindow
from ui_file.login import Ui_Login
from ui_file.register import Ui_Register
from ui_file.query import Ui_Query
import sqlite3
import time
from PIL import Image
import onnxruntime as ort
import base64

class Main(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)
        self.showMaximized()
        self.setStyleSheet("#MainWindow{border-image:url(background/veer-160992005.jpg);}")
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setFilter(QtCore.QDir.Dirs | QtCore.QDir.NoDotAndDotDot)
        self.fileSystemModel.setRootPath('.')
        self.treeView.setModel(self.fileSystemModel)

        self.treeView.setColumnWidth(0, 500)
        self.treeView.setColumnHidden(1, True)
        self.treeView.setColumnHidden(2, True)
        self.treeView.setColumnHidden(3, True)
        self.treeView.header().hide()
        self.treeView.doubleClicked.connect(self.select_image)

        self.listWidget.setViewMode(QListView.IconMode)
        self.listWidget.setSpacing(10)
        self.listWidget.setIconSize(QSize(150, 100))
        self.listWidget.setMovement(False)
        self.listWidget.itemDoubleClicked.connect(self.item_click)
        self.listWidget.setResizeMode(QListView.Adjust)
        self.progressBar.hide()

        self.graphicsView.setAlignment(Qt.AlignCenter)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.checkBox.stateChanged.connect(self.maofa)
        self.checkBox_2.stateChanged.connect(self.fenge)
        self.checkBox_3.stateChanged.connect(self.pinghua)
        self.checkBox_4.stateChanged.connect(self.gama)
        self.checkBox_5.stateChanged.connect(self.junhenghua)
        self.checkBox_6.stateChanged.connect(self.junhenghua)
        self.checkBox_7.stateChanged.connect(self.junhenghua)
        self.checkBox_8.stateChanged.connect(self.junhenghua)
        self.checkBox_9.stateChanged.connect(self.liangdu)
        self.checkBox_10.stateChanged.connect(self.bianyuan)
        self.checkBox_11.stateChanged.connect(self.zaosheng)
        self.checkBox_12.stateChanged.connect(self.zaosheng)
        self.checkBox_13.stateChanged.connect(self.zaosheng)

        self.operate = []
        self.image = None
        self.pro_image = None

        self.spinBox.setMinimum(3)
        self.spinBox.setMaximum(9)
        self.spinBox.setSingleStep(1)
        self.spinBox.setValue(3)
        self.spinBox.valueChanged.connect(self.pinghuavaluechange)

        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(1000)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setValue(100)
        self.horizontalSlider.valueChanged.connect(self.gamavaluechange)
        self.lineEdit.setText(str(round(self.horizontalSlider.value() * 0.01, 2)))

        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(300)
        self.horizontalSlider_2.setSingleStep(1)
        self.horizontalSlider_2.setValue(100)
        self.horizontalSlider_2.valueChanged.connect(self.liangduvaluechange)
        self.lineEdit_2.setText(str(round(self.horizontalSlider_2.value() * 0.01, 2)))

        self.horizontalSlider_3.setMinimum(0)
        self.horizontalSlider_3.setMaximum(1000)
        self.horizontalSlider_3.setSingleStep(1)
        self.horizontalSlider_3.setValue(100)
        self.horizontalSlider_3.valueChanged.connect(self.liangduvaluechange)
        self.lineEdit_3.setText(str(round(self.horizontalSlider_3.value() * 0.01, 2)))

        self.horizontalSlider_4.setMinimum(0)
        self.horizontalSlider_4.setMaximum(1000)
        self.horizontalSlider_4.setSingleStep(1)
        self.horizontalSlider_4.setValue(100)
        self.horizontalSlider_4.valueChanged.connect(self.bianyuanvaluechange)
        self.lineEdit_4.setText(str(round(self.horizontalSlider_4.value() * 0.1, 2)))

        self.horizontalSlider_5.setMinimum(0)
        self.horizontalSlider_5.setMaximum(1000)
        self.horizontalSlider_5.setSingleStep(1)
        self.horizontalSlider_5.setValue(100)
        self.horizontalSlider_5.valueChanged.connect(self.bianyuanvaluechange)
        self.lineEdit_5.setText(str(round(self.horizontalSlider_5.value() * 0.1, 2)))

        self.pushButton_2.clicked.connect(self.chaxun)
        self.pushButton.clicked.connect(self.startrecognition)

        self.pushButton_3.clicked.connect(self.slotStart)
        self.pushButton_4.clicked.connect(self.slotStop)
        self.pushButton_5.clicked.connect(self.photocap)

        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.openFrame)

        if not os.path.exists('database/cancer.db'):
            self.initdb()

    def initdb(self):
        conn = sqlite3.connect('database/cancer.db')
        cursor = conn.cursor()
        cursor.execute(
            'create table user(result varchar(90),date varchar(20),time varchar(20),detect_time varchar(20),image blob)')
        cursor.close()
        conn.close()

    def photocap(self):
        cv2.imwrite('./cap_img/cap_img.jpg',self.frame)
        self.cap_photo = self.frame
        self.cap_photo = cv2.cvtColor(self.cap_photo, cv2.COLOR_BGR2RGB)
        self.cap.release()
        self.timer.exit()
        self.scene.removeItem(self.item)
        width = int((self.graphicsView.height() / self.cap_photo.shape[0]) * self.cap_photo.shape[1])
        image = cv2.resize(self.cap_photo, (width, self.graphicsView.height()))
        x = image.shape[1]
        y = image.shape[0]
        showImage_1 = QtGui.QImage(image.data, x, y, x * 3,
                                   QtGui.QImage.Format_RGB888)
        self.item = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(showImage_1))
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

        self.detection_path = './cap_img/cap_img.jpg'
        self.image = cv2.imdecode(np.fromfile(self.detection_path, dtype=np.uint8), -1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def slotStart(self):
        self.textEdit.clear()
        self.cap = cv2.VideoCapture(0)
        self.timer.start()

    def slotStop(self):
        self.image = None
        self.cap.release()
        self.timer.exit()
        self.scene.removeItem(self.item)

    def openFrame(self):
        ret, self.frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            width = int((self.graphicsView.height() / self.frame.shape[0]) * self.frame.shape[1])
            image = cv2.resize(image, (width, self.graphicsView.height()))
            x = image.shape[1]
            y = image.shape[0]
            showImage_4 = QtGui.QImage(image.data, x, y, x * 3,
                                       QtGui.QImage.Format_RGB888)
            self.item = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(showImage_4))
            self.scene = QGraphicsScene()
            self.scene.addItem(self.item)
            self.graphicsView.setScene(self.scene)
        else:
            self.cap.release()

    def startrecognition(self):
        if self.image is None:
            QMessageBox.warning(self, "警告", "输入图片不能为空!")
        else:
            self.classes = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
            self.mbt = DetectThread(self.detection_path,self.classes)
            self.mbt.trigger.connect(self.recognition)
            self.mbt.start()

    def recognition(self,result):
        self.textEdit.setText(result)

    def chaxun(self):
        self.chaxun_gui = Query()
        self.chaxun_gui.show()

    def fenge(self):
        if self.checkBox_2.isChecked():
            if 'fenge' not in self.operate:
                self.operate.append('fenge')
        else:
            if 'fenge' in self.operate:
                self.operate.remove('fenge')
        self.process_fun()

    def bianyuan(self):
        if self.checkBox_10.isChecked():
            if 'bianyuan' not in self.operate:
                self.operate.append('bianyuan')
        else:
            if 'bianyuan' in self.operate:
                self.operate.remove('bianyuan')
        self.process_fun()

    def bianyuanvaluechange(self):
        self.lineEdit_4.setText(str(round(self.horizontalSlider_4.value() * 0.01, 2)))
        self.lineEdit_5.setText(str(round(self.horizontalSlider_5.value() * 0.01, 2)))
        if self.checkBox_10.isChecked():
            self.process_fun()

    def liangdu(self):
        if self.checkBox_9.isChecked():
            if 'liangdu' not in self.operate:
                self.operate.append('liangdu')
        else:
            if 'liangdu' in self.operate:
                self.operate.remove('liangdu')
        self.process_fun()

    def liangduvaluechange(self):
        self.lineEdit_2.setText(str(round(self.horizontalSlider_2.value() * 0.01, 2)))
        self.lineEdit_3.setText(str(round(self.horizontalSlider_3.value() * 0.01, 2)))
        if self.checkBox_9.isChecked():
            self.process_fun()

    def junhenghua(self):
        if self.checkBox_5.isChecked():
            if 'junhenghua' not in self.operate:
                self.operate.append('junhenghua')
        else:
            if 'junhenghua' in self.operate:
                self.operate.remove('junhenghua')
        self.process_fun()

    def gama(self):
        if self.checkBox_4.isChecked():
            if 'gama' not in self.operate:
                self.operate.append('gama')
        else:
            if 'gama' in self.operate:
                self.operate.remove('gama')
        self.process_fun()

    def gamavaluechange(self):
        self.lineEdit.setText(str(round(self.horizontalSlider.value() * 0.01, 2)))
        if self.checkBox_4.isChecked():
            self.process_fun()

    def zaosheng(self):
        if self.checkBox_11.isChecked():
            if 'zaosheng' not in self.operate:
                self.operate.append('zaosheng')
        else:
            if 'zaosheng' in self.operate:
                self.operate.remove('zaosheng')
        self.process_fun()


    def maofa(self):
        if self.checkBox.isChecked():
            if 'maofa' not in self.operate:
                self.operate.append('maofa')
        else:
            if 'maofa' in self.operate:
                self.operate.remove('maofa')
        self.process_fun()

    def pinghua(self):
        if self.checkBox_3.isChecked():
            if 'pinghua' not in self.operate:
                self.operate.append('pinghua')
        else:
            if 'pinghua' in self.operate:
                self.operate.remove('pinghua')
        self.process_fun()

    def pinghuavaluechange(self):
        if self.checkBox_3.isChecked():
            self.process_fun()

    def contextMenuEvent(self, event):
        menu = QMenu()
        save_action = QAction('另存为', self)
        save_action.triggered.connect(self.save_current)
        menu.addAction(save_action)
        menu.exec(QCursor.pos())

    def save_current(self):
        file_name = QFileDialog.getSaveFileName(self, '另存为', './', 'Image files(*.jpg *.gif *.png)')[0]
        print(file_name)
        if file_name:
            self.item.pixmap().save(file_name)

    def process_fun(self):
        if self.image is not None:
            self.pro_image = self.image.copy()
            for i in self.operate:
                if i == 'maofa':
                    grayScale = cv2.cvtColor(self.pro_image, cv2.COLOR_RGB2GRAY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
                    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
                    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
                    self.pro_image = cv2.inpaint(self.pro_image, thresh2, 1, cv2.INPAINT_TELEA)

                if i == 'pinghua':
                    self.pro_image = cv2.blur(self.pro_image, (self.spinBox.value(), self.spinBox.value()))

                if i == 'zaosheng':
                    if self.checkBox_12.isChecked():
                        image = np.array(self.pro_image / 255, dtype=float)
                        noise = np.random.normal(0, 25 / 255.0, image.shape)
                        out = image + noise
                        res_img = np.clip(out, 0.0, 1.0)
                        self.pro_image = np.uint8(res_img * 255.0)
                    if self.checkBox_13.isChecked():
                        noisy_img = np.copy(self.pro_image)
                        num_salt = np.ceil(0.04 * self.pro_image.size * 0.5)
                        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.pro_image.shape]
                        noisy_img[coords[0], coords[1], :] = [255, 255, 255]
                        num_pepper = np.ceil(0.04 * self.pro_image.size * (1. - 0.5))
                        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.pro_image.shape]
                        noisy_img[coords[0], coords[1], :] = [0, 0, 0]
                        self.pro_image = noisy_img

                if i == 'gama':
                    gamma_table = [np.power(x / 255.0, self.horizontalSlider.value() * 0.01) * 255.0 for x in
                                   range(256)]
                    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
                    self.pro_image = cv2.LUT(self.pro_image, gamma_table)
                    self.lineEdit.setText(str(round(self.horizontalSlider.value() * 0.01, 2)))

                if i == 'junhenghua':
                    b, g, r = cv2.split(self.pro_image)
                    if self.checkBox_8.isChecked():
                        b = cv2.equalizeHist(b)
                    if self.checkBox_7.isChecked():
                        g = cv2.equalizeHist(g)
                    if self.checkBox_6.isChecked():
                        r = cv2.equalizeHist(r)
                    self.pro_image = cv2.merge((b, g, r))

                if i == 'liangdu':
                    blank = np.zeros(self.pro_image.shape, self.pro_image.dtype)
                    self.pro_image = cv2.addWeighted(self.pro_image, self.horizontalSlider_2.value() * 0.01, blank,
                                          1 - self.horizontalSlider_2.value() * 0.01,
                                          self.horizontalSlider_3.value() * 0.01)

                if i == 'bianyuan':
                    self.pro_image = cv2.Canny(self.pro_image, threshold1=self.horizontalSlider_4.value() * 0.1,
                                    threshold2=self.horizontalSlider_5.value() * 0.1)
                    self.pro_image = cv2.cvtColor(self.pro_image, cv2.COLOR_GRAY2BGR)
                    self.lineEdit_4.setText(str(round(self.horizontalSlider_4.value() * 0.1, 2)))
                    self.lineEdit_5.setText(str(round(self.horizontalSlider_5.value() * 0.1, 2)))

                if i == 'fenge':
                    pr_img = self.pro_image.copy()
                    self.seg = SegThread(self.detection_path, pr_img)
                    self.seg.trigger.connect(self.fengeimg)
                    self.seg.start()

            width = int((self.graphicsView.height() / self.image.shape[0]) * self.image.shape[1])
            image = cv2.resize(self.pro_image, (width, self.graphicsView.height()))
            x = image.shape[1]
            y = image.shape[0]
            showImage1 = QtGui.QImage(image.data, x, y, x * 3,
                                       QtGui.QImage.Format_RGB888)
            self.item = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(showImage1))
            self.scene = QGraphicsScene()
            self.scene.addItem(self.item)
            self.graphicsView.setScene(self.scene)
        else:
            QMessageBox.warning(self, "警告", "输入图片不能为空!")

    def fengeimg(self,x):

        ret_p, thresh_p = cv2.threshold(x, 127, 255, 0)
        contours_p, hierarchy_p = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        overlap_img = cv2.drawContours(self.pro_image, contours_p, -1, (255, 36, 0), 8)
        width = int((self.graphicsView.height() / self.image.shape[0]) * self.image.shape[1])
        image = cv2.resize(overlap_img,(width, self.graphicsView.height()))
        x = image.shape[1]
        y = image.shape[0]
        showImage1 = QtGui.QImage(image.data, x, y, x * 3,
                                  QtGui.QImage.Format_RGB888)
        self.item = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(showImage1))
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

    def select_image(self, file_index):
        if self.tabWidget.currentIndex() == 0:
            self.listWidget.clear()
            self.file_path = self.fileSystemModel.filePath(file_index)
            files = os.listdir(self.file_path)
            process = 0
            for f1 in files:
                process += 1
                value = int((process / len(files)) * 100)
                self.progressBar.show()
                self.progressBar.setProperty("value", value)
                if value == 100:
                    self.progressBar.hide()
                if f1.endswith(('.jpg', '.png', '.bmp')):
                    item = QtWidgets.QListWidgetItem(QtGui.QIcon(os.path.join(self.file_path, f1)),
                                                     os.path.split(f1)[-1])
                    self.listWidget.addItem(item)

    def item_click(self,item):
        self.textEdit.clear()
        self.detection_path = os.path.join(self.file_path,item.text())
        self.image = cv2.imdecode(np.fromfile(self.detection_path, dtype=np.uint8), -1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        width = int((self.graphicsView.height()/self.image.shape[0])*self.image.shape[1])
        image = cv2.resize(self.image,(width,self.graphicsView.height()))
        x = image.shape[1]
        y = image.shape[0]
        showImage = QtGui.QImage(image.data, x, y, x*3,
                                   QtGui.QImage.Format_RGB888)
        self.item = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(showImage))
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)


class Communicate(QObject):
    signal = pyqtSignal(str)
class VideoTimer(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.timeSignal = Communicate()

    def run(self):
        while True:
            self.timeSignal.signal.emit("1")
            time.sleep(1 / 20)

class DetectThread(QThread):
    trigger = pyqtSignal(str)

    def __init__(self,path,detclass):
        super(DetectThread, self).__init__()

        self.detection_path = path
        self.classes = detclass

        self.PIXEL_MEANS = (0.485, 0.456, 0.406)
        self.PIXEL_STDS = (0.229, 0.224, 0.225)

        self.sess = ort.InferenceSession("./weights/cls.onnx")

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def run(self):

        img = Image.open(self.detection_path).convert('RGB')
        img = img.resize((600,450))
        img = np.array(img)
        img = img.astype(np.float32, copy=False)
        img /= 255.0
        img -= np.array(self.PIXEL_MEANS)
        img /= np.array(self.PIXEL_STDS)
        input_data = np.transpose(img, (2, 0, 1))
        input_data = input_data.reshape([1, 3, 450, 600])

        start_time = time.time()
        input_name = self.sess.get_inputs()[0].name
        result = self.sess.run([], {input_name: input_data})

        pre_label = self.softmax(np.squeeze(result[0]))
        index = np.argmax(pre_label)

        end_time = time.time()

        detec_time = round((end_time - start_time),2)

        result = self.classes[index]+' '+str(round(pre_label[index]*100,1))+'%'

        save_image = cv2.imread(self.detection_path)
        retval, encoded_image = cv2.imencode(".jpg", save_image)
        content = base64.b64encode(encoded_image)

        conn = sqlite3.connect('database/cancer.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO user VALUES (?,?,?,?,?)', (
        result,str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())).split(' ')[0],
        str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())).split(' ')[1],detec_time,content))
        cursor.close()
        conn.commit()
        conn.close()

        self.trigger.emit(result)

class SegThread(QThread):
    trigger = pyqtSignal(np.ndarray)

    def __init__(self,path,process_img):
        super(SegThread, self).__init__()

        self.detection_path = path
        self.result_img = process_img

        self.PIXEL_MEANS = (0.485, 0.456, 0.406)
        self.PIXEL_STDS = (0.229, 0.224, 0.225)

        self.sess = ort.InferenceSession("./weights/seg.onnx")

    def run(self):

        ori_image = Image.open(self.detection_path).convert('RGB')
        img = ori_image.resize((512,512))
        img = np.array(img)
        img = img.astype(np.float32, copy=False)
        img /= 255.0
        img -= np.array(self.PIXEL_MEANS)
        img /= np.array(self.PIXEL_STDS)
        input_data = np.transpose(img, (2, 0, 1))
        input_data = input_data.reshape([1, 3, 512, 512])
        input_name = self.sess.get_inputs()[0].name
        result = self.sess.run([], {input_name: input_data})
        prediction = np.squeeze(result).argmax(axis=0)
        prediction[prediction==0] = 0.0
        prediction[prediction==1] = 255.0
        prediction = prediction.astype("uint8")
        prediction = cv2.resize(prediction, (ori_image.size[0], ori_image.size[1]))

        self.trigger.emit(prediction)

class Login(QWidget,Ui_Login):
    def __init__(self):
        super(Login,self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.zhuce)
        self.pushButton.clicked.connect(self.denglu)
        self.attach = 0
        if not os.path.exists('database/user.db'):
            self.initdb()

    def initdb(self):
        conn = sqlite3.connect('database/user.db')
        cursor = conn.cursor()
        cursor.execute(
            'create table user(name varchar(50),result varchar(50))')
        cursor.execute('INSERT INTO user VALUES (?,?)', ('yao', '123'))
        cursor.close()
        conn.commit()
        conn.close()

    def zhuce(self):
        self.zhuce = Register()
        self.zhuce.show()

    def denglu(self):
        conn = sqlite3.connect('./database/user.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user')
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        account = self.lineEdit.text()
        password = self.lineEdit_2.text()
        if account == "" or password == "":
            QMessageBox.warning(self, "警告", "账号密码不能为空，请输入！")
            return
        else:
            for j in result:
                if account == str(j[0]) and password == str(j[1]):
                    self.attach = 1
                    break
            if self.attach == 1:
                self.main = Main()
                self.main.show()
                self.close()
            else:
                QMessageBox.warning(self, "警告", "账户或密码错误，请重新输入！")


class Register(QWidget,Ui_Register):
    def __init__(self):
        super(Register,self).__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.queding)
        self.pushButton_2.clicked.connect(self.quit)

    def queding(self):
        self.account = self.lineEdit.text()
        self.password = self.lineEdit_2.text()
        self.re_password = self.lineEdit_3.text()

        if self.account == "" or self.password == "" or self.re_password == "":
            QMessageBox.warning(self, "警告", "请输入用户名或密码！")

        elif self.password != self.re_password:
            QMessageBox.warning(self, "警告", "输入密码不一致，请重新输入！")
        else:
            conn = sqlite3.connect('./database/user.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO user VALUES (?,?)', (self.account, self.password))
            cursor.close()
            conn.commit()
            conn.close()
            QMessageBox.warning(self, "成功", "注册成功！")

    def quit(self):
        self.close()

class Query(QWidget,Ui_Query):
    def __init__(self):
        super(Query,self).__init__()
        self.setupUi(self)
        self.showMaximized()
        self.desktop = QApplication.desktop()
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("background/query.jpg").scaled(self.desktop.width(), self.desktop.height())))
        self.setPalette(palette)
        self.tableWidget.setHorizontalHeaderLabels(['日期','时间','识别结果','识别用时(s)'])
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.verticalHeader().setDefaultSectionSize(110)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.tableWidget_2.verticalHeader().setVisible(False)
        self.tableWidget_2.horizontalHeader().setVisible(False)
        self.tableWidget_2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget_2.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget_2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget_2.setItem(0, 0, QTableWidgetItem('AKIEC'))
        self.tableWidget_2.setItem(1, 0, QTableWidgetItem('BCC'))
        self.tableWidget_2.setItem(2, 0, QTableWidgetItem('BKL'))
        self.tableWidget_2.setItem(3, 0, QTableWidgetItem('DF'))
        self.tableWidget_2.setItem(4, 0, QTableWidgetItem('MEL'))
        self.tableWidget_2.setItem(5, 0, QTableWidgetItem('NV'))
        self.tableWidget_2.setItem(6, 0, QTableWidgetItem('VASC'))
        self.tableWidget_2.doubleClicked.connect(self.itemClicked)

        self.label_5.setAlignment(Qt.AlignCenter)
        self.label_6.setAlignment(Qt.AlignCenter)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.tableWidget.doubleClicked.connect(self.record_check)
        self.calendarWidget.clicked.connect(self.show_riqi)

    def record_check(self,index):
        table_row = index.row()
        date = self.tableWidget.item(table_row, 0).text()
        time = self.tableWidget.item(table_row, 1).text()
        detect_result = self.tableWidget.item(table_row, 2).text()
        detect_time = self.tableWidget.item(table_row, 3).text()

        conn = sqlite3.connect('database/cancer.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user')
        result = cursor.fetchall()
        cursor.close()
        conn.close()

        for i in result:
            if i[0] == detect_result and i[1] == date and i[2] == time and i[3] == detect_time:
                img_data = base64.b64decode(i[4])
                pixmap = QtGui.QPixmap()
                pixmap.loadFromData(img_data)
                scaled_pixmap = pixmap.scaled(self.label_2.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.label_2.setPixmap(scaled_pixmap)

    def show_riqi(self):
        show_day = []
        self.select_day = self.calendarWidget.selectedDate().toString("yyyy-MM-dd")
        conn = sqlite3.connect('database/cancer.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user')
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        for j in range(len(result)):
            if str(result[j][1]) == self.select_day:
                show_day.append(result[j])
        self.tableWidget.setRowCount(len(show_day))
        for i in range(len(show_day)):
            show_every = show_day[i]
            self.tableWidget.setItem(i, 0, QTableWidgetItem(str(show_every[1])))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(show_every[2])))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(show_every[0])))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(show_every[3])))

    def itemClicked(self,index):
        name_row = index.row()
        text = self.tableWidget_2.item(name_row, 0).text()
        if text == 'AKIEC':
            self.textEdit.setText('光化性角化病(AKIEC)是一种职业病，主要受日光、紫外线、放射性热能以及沥青或煤及其提炼而物诱发本病。病损多见于中年以上男性日光暴露部位，如面部、耳廓、手背等。主要表现为表面粗糙，可见角化性鳞屑。揭去鳞屑，可见下方的基面红润，凹凸不平，呈乳头状。治疗一般采取外用药和手术治疗。有20%可继发鳞癌。')
            pixmap = QPixmap('./example/akiec1.jpg')
            scaled_pixmap = pixmap.scaled(self.label_5.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_5.setPixmap(scaled_pixmap)

            pixmap = QPixmap('./example/akiec2.jpg')
            scaled_pixmap = pixmap.scaled(self.label_6.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_6.setPixmap(scaled_pixmap)


        if text == 'BCC':
            self.textEdit.setText('基底细胞癌(BCC)发生转移率低，比较偏向于良性，故又称基底细胞上皮瘤。基于它有较大的破坏性，又称侵袭性溃疡。基底细胞癌多见于老年人，好发于头、面、颈及手背等处，尤其是面部较突出的部位。开始是一个皮肤色到暗褐色浸润的小结节，较典型者为蜡样、半透明状结节，有高起卷曲的边缘。中央开始破溃，结黑色坏死性痂，中心坏死向深部组织扩展蔓延，呈大片状侵袭性坏死，可以深达软组织和骨组织。')
            pixmap = QPixmap('./example/bcc1.jpg')
            scaled_pixmap = pixmap.scaled(self.label_5.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_5.setPixmap(scaled_pixmap)

            pixmap = QPixmap('./example/bcc2.jpg')
            scaled_pixmap = pixmap.scaled(self.label_6.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_6.setPixmap(scaled_pixmap)

        if text == 'BKL':
            self.textEdit.setText('扁平苔藓样角化病(BKL)是指有苔藓样组织学改变的良性角化性皮肤病，又称苔藓样角化病、良性苔藓样角化病。好发于中老年人的日光暴露部位，表现为单发的边界清楚的角化性斑，自觉症状不一。')
            pixmap = QPixmap('./example/bkl1.jpg')
            scaled_pixmap = pixmap.scaled(self.label_5.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_5.setPixmap(scaled_pixmap)

            pixmap = QPixmap('./example/bkl2.jpg')
            scaled_pixmap = pixmap.scaled(self.label_6.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_6.setPixmap(scaled_pixmap)

        if text == 'DF':
            self.textEdit.setText('皮肤纤维瘤(DF)是成纤维细胞或组织细胞灶性增生引致的一种真皮内的良性肿瘤。本病可发生于任何年龄，中青年多见，女性多于男性。可自然发生或外伤后引起。黄褐色或淡红色的皮内丘疹或结节是本病的临床特征。病损生长缓慢，长期存在，极少自行消退。')
            pixmap = QPixmap('./example/df1.jpg')
            scaled_pixmap = pixmap.scaled(self.label_5.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_5.setPixmap(scaled_pixmap)

            pixmap = QPixmap('./example/df2.jpg')
            scaled_pixmap = pixmap.scaled(self.label_6.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_6.setPixmap(scaled_pixmap)

        if text == 'MEL':
            self.textEdit.setText('黑色素瘤(MEL)，通常是指恶性黑色素瘤，是黑色素细胞来源的一种高度恶性的肿瘤，简称恶黑，多发生于皮肤，也可见于黏膜和内脏，约占全部肿瘤的3%。皮肤恶性黑色素瘤占皮肤恶性肿瘤的第三位（约占6.8%~20%）。好发于成人，皮肤白皙的白种人发病率高，而深色皮肤的亚洲人和非洲人发病率较低，极少见于儿童。部分患者有家族性多发现象。')
            pixmap = QPixmap('./example/mel1.jpg')
            scaled_pixmap = pixmap.scaled(self.label_5.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_5.setPixmap(scaled_pixmap)

            pixmap = QPixmap('./example/mel2.jpg')
            scaled_pixmap = pixmap.scaled(self.label_6.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_6.setPixmap(scaled_pixmap)

        if text == 'NV':
            self.textEdit.setText('黑色素痣(NV)是由一群良性的黑色素细胞，聚集在表皮与真皮的交界产生的。黑色素细胞可能会分布在网状真皮下部（lower reticular diemis），结缔组织束之间（between collagen bundles），围绕皮肤的其它附属器官如汗腺、毛囊、血管、神经等等，偶尔还会延伸到皮下脂肪。')
            pixmap = QPixmap('./example/nv1.jpg')
            scaled_pixmap = pixmap.scaled(self.label_5.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_5.setPixmap(scaled_pixmap)

            pixmap = QPixmap('./example/nv2.jpg')
            scaled_pixmap = pixmap.scaled(self.label_6.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_6.setPixmap(scaled_pixmap)

        if text == 'VASC':
            self.textEdit.setText('血管性皮肤病(VASC)是指原发于皮肤血管管壁的一类炎症性疾病，其共同组织病理表现为血管内皮细胞肿胀，血管壁纤维蛋白样变性及管周炎症细胞浸润或肉芽肿形成。')
            pixmap = QPixmap('./example/vasc1.jpg')
            scaled_pixmap = pixmap.scaled(self.label_5.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_5.setPixmap(scaled_pixmap)

            pixmap = QPixmap('./example/vasc2.jpg')
            scaled_pixmap = pixmap.scaled(self.label_6.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_6.setPixmap(scaled_pixmap)



if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    ui = Login()
    ui.show()

    sys.exit(app.exec_())