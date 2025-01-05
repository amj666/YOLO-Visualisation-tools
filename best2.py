import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QMessageBox, QFileDialog, QMenu
from PyQt5.QtGui import QImage, QPixmap, QClipboard
import cv2
from ultralytics import YOLO


class Worker:
    def __init__(self):
        self.model = None

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(None, "选择模型文件", "", "模型文件 (*.pt)")
        if model_path:
            self.model = YOLO(model_path)
            return self.model is not None
        return False

    def detect_image(self, image):
        results = self.model.predict(image)
        return results


class InteractiveLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_pixmap = None  # 用于存储当前显示的图片

    def set_image(self, pixmap):
        """设置当前图片并存储它"""
        self.image_pixmap = pixmap
        self.setPixmap(pixmap)

    def contextMenuEvent(self, event):
        """右键菜单事件"""
        if self.image_pixmap is not None:
            menu = QMenu(self)

            # 设置菜单的样式表
            menu.setStyleSheet("""
                QMenu {
                    background-color: white;
                    border: 1px solid lightgray;
                }
                QMenu::item {
                    padding: 6px 20px;
                    background-color: transparent;
                }
                QMenu::item:selected {
                    background-color: #6950a1;
                    color: white;
                }
            """)

            save_action = menu.addAction("保存图片")
            copy_action = menu.addAction("复制图片")
            action = menu.exec_(self.mapToGlobal(event.pos()))
            if action == save_action:
                self.save_image()
            elif action == copy_action:
                self.copy_image()

    def save_image(self):
        """保存图片到本地"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "图片文件 (*.png *.jpg *.bmp)")
        if file_path:
            self.image_pixmap.save(file_path)
            QMessageBox.information(self, "保存成功", f"图片已保存到 {file_path}")

    def copy_image(self):
        """复制图片到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setPixmap(self.image_pixmap)  # 复制图片到剪贴板


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片与视频检测程序V2.1——By AMJ")
        
        # 默认图片最小尺寸
        self.image_min_width = 580
        self.image_min_height = 450

        # 根据图片尺寸动态调整窗口最小尺寸
        self.setMinimumSize(self.image_min_width * 2 + 60, self.image_min_height + 100)

        self.setGeometry(300, 150, 1000, 600)

        # 创建两个 InteractiveLabel 分别显示左右图像
        self.label1 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumSize(self.image_min_width, self.image_min_height)
        self.label1.setStyleSheet('border:3px solid #6950a1; background-color: black;')

        self.label2 = InteractiveLabel()
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumSize(self.image_min_width, self.image_min_height)
        self.label2.setStyleSheet('border:3px solid #6950a1; background-color: black;')

        # 水平布局，用于放置左右两个 QLabel
        layout = QVBoxLayout()
        hbox_video = QHBoxLayout()
        hbox_video.addWidget(self.label1)  # 左侧显示原始图像
        hbox_video.addWidget(self.label2)  # 右侧显示检测后的图像
        layout.addLayout(hbox_video)
        self.worker = Worker()

        # 创建按钮布局
        hbox_buttons = QHBoxLayout()

        # 添加模型选择按钮
        self.load_model_button = QPushButton("📁模型选择")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.load_model_button)

        # 添加导入图片按钮
        self.load_images_button = QPushButton("🖼️导入图片")
        self.load_images_button.clicked.connect(self.load_images)
        self.load_images_button.setEnabled(False)
        self.load_images_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.load_images_button)

        # 添加导入视频按钮
        self.load_video_button = QPushButton("🎞️导入视频")
        self.load_video_button.clicked.connect(self.load_video)
        self.load_video_button.setEnabled(False)
        self.load_video_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.load_video_button)

        # 添加上一张按钮
        self.prev_image_button = QPushButton("◀上一张")
        self.prev_image_button.clicked.connect(self.show_prev_image)
        self.prev_image_button.setEnabled(False)
        self.prev_image_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.prev_image_button)

        # 添加下一张按钮
        self.next_image_button = QPushButton("▶下一张")
        self.next_image_button.clicked.connect(self.show_next_image)
        self.next_image_button.setEnabled(False)
        self.next_image_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.next_image_button)

        # 添加停止按钮
        self.stop_button = QPushButton("⏹️停止")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.stop_button)

        # 添加显示检测物体按钮
        self.display_objects_button = QPushButton("🔍统计")
        self.display_objects_button.clicked.connect(self.show_detected_objects)
        self.display_objects_button.setEnabled(False)
        self.display_objects_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.display_objects_button)

        # 添加退出按钮
        self.exit_button = QPushButton("❌退出")
        self.exit_button.clicked.connect(self.exit_application)
        self.exit_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.exit_button)

        layout.addLayout(hbox_buttons)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.current_results = None
        self.image_paths = []  # 存储图片路径
        self.current_image_index = -1  # 当前图片索引

        # 视频播放变量
        self.video_path = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.video_play)

        # 当前图片数据
        self.original_image = None
        self.annotated_image = None
# -------------------------------------------
    def load_model(self):
        if self.worker.load_model():
            self.load_images_button.setEnabled(True)
            self.load_video_button.setEnabled(True)
            self.display_objects_button.setEnabled(True)
            self.stop_button.setEnabled(True)

    def load_images(self):
        """导入多张图片"""
        file_names, _ = QFileDialog.getOpenFileNames(self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if file_names:
            self.image_paths = file_names
            self.current_image_index = 0
            self.show_current_image()
            self.prev_image_button.setEnabled(len(self.image_paths) > 1)
            self.next_image_button.setEnabled(len(self.image_paths) > 1)

    def load_video(self):
        """导入视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.timer.start(60)  # 每 60 毫秒播放一帧
            self.stop_button.setEnabled(True)

    def video_play(self):
        """播放视频并检测"""
        if self.video_capture is None or not self.video_capture.isOpened():
            self.timer.stop()
            return

        ret, frame = self.video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.worker.model is not None:
                results = self.worker.detect_image(frame_rgb)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # 显示原始帧
                height1, width1, channel1 = frame_rgb.shape
                bytesPerLine1 = 3 * width1
                qimage1 = QImage(frame_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
                pixmap1 = QPixmap.fromImage(qimage1)
                self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio))

                # 显示检测后的帧
                height2, width2, channel2 = annotated_frame_rgb.shape
                bytesPerLine2 = 3 * width2
                qimage2 = QImage(annotated_frame_rgb.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
                pixmap2 = QPixmap.fromImage(qimage2)
                self.label2.set_image(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio))
        else:
            self.video_capture.release()
            self.timer.stop()

    def resizeEvent(self, event):
        """当窗口大小发生变化时，重新加载图片以防止图片变花"""
        if self.original_image is not None and self.annotated_image is not None:
            self.show_images(self.original_image, self.annotated_image)
# -------------------------------------------
    def show_images(self, original, annotated):
        """显示原始和检测后的图片"""
        # 显示原始图片
        image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        height1, width1, channel1 = image_rgb.shape
        bytesPerLine1 = 3 * width1
        qimage1 = QImage(image_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
        pixmap1 = QPixmap.fromImage(qimage1)
        self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio))

        # 显示检测后的图片
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        height2, width2, channel2 = annotated_rgb.shape
        bytesPerLine2 = 3 * width2
        qimage2 = QImage(annotated_rgb.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
        pixmap2 = QPixmap.fromImage(qimage2)
        self.label2.set_image(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio))

    def show_current_image(self):
        """显示当前选定的图片"""
        if 0 <= self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            self.original_image = cv2.imread(image_path)
            if self.original_image is not None:
                self.current_results = self.worker.detect_image(self.original_image)
                if self.current_results:
                    self.annotated_image = self.current_results[0].plot()
                    self.show_images(self.original_image, self.annotated_image)
    
    def show_prev_image(self):
        """显示上一张图片"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()
    
    def show_next_image(self):
        """显示下一张图片"""
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.show_current_image()
# -------------------------------------------   
    def show_detected_objects(self):
        if self.current_results:
            det_info = self.current_results[0].boxes.cls
            object_count = len(det_info)
            object_info = f"识别到的物体总个数：{object_count}\n"
            object_dict = {}
            class_names_dict = self.current_results[0].names
            for class_id in det_info:
                class_name = class_names_dict[int(class_id)]
                if class_name in object_dict:
                    object_dict[class_name] += 1
                else:
                    object_dict[class_name] = 1
            sorted_objects = sorted(object_dict.items(), key=lambda x: x[1], reverse=True)
            for obj_name, obj_count in sorted_objects:
                object_info += f"{obj_name}: {obj_count}\n"
            self.show_message_box("识别结果", object_info)
        else:
            self.show_message_box("识别结果", "未检测到物体")
    
    def show_message_box(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()
# -------------------------------------------
    def stop_processing(self):
        """停止视频播放或其他处理"""
        if self.timer.isActive():
            self.timer.stop()
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
        self.label1.clear()
        self.label2.clear()
        self.stop_button.setEnabled(False)

    def exit_application(self):
        sys.exit()

# -------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
