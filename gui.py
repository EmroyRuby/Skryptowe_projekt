from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton, QGridLayout, QGroupBox, \
    QHBoxLayout, QVBoxLayout, QRadioButton
from PyQt6.QtGui import QPixmap
import os
import sys
import model


class Gui(QApplication):
    def __init__(self):
        super().__init__([])
        self.window = Window()

    def run(self):
        self.window.start()
        sys.exit(self.exec())


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Rice image recognition')
        self.setGeometry(300, 300, 400, 200)

        grid = QGridLayout()

        group1 = QGroupBox('Choose a picture:')
        hbox = QHBoxLayout(self)
        group1.setLayout(hbox)

        group2 = QGroupBox('Choose a model:')
        vbox = QVBoxLayout(self)
        group2.setLayout(vbox)

        self.pic_label = QLabel(self)

        button1 = QPushButton('Browser...', self)
        button1.resize(button1.sizeHint())
        button1.clicked.connect(self.load_pic)
        hbox.addWidget(button1)

        self.text_label = QLabel(self)
        hbox.addWidget(self.text_label)

        self.rad_button_1 = QRadioButton('Default')
        self.rad_button_1.setChecked(True)
        self.rad_button_1.clicked.connect(self.set_default_model)
        vbox.addWidget(self.rad_button_1)

        self.rad_button_2 = QGroupBox('Choose my own:')
        self.rad_button_2.setCheckable(True)
        self.rad_button_2.setChecked(False)
        self.rad_button_2.setFlat(True)
        self.rad_button_2.clicked.connect(self.rad_button_2_check)
        hbox_2 = QHBoxLayout(self)
        self.rad_button_2.setLayout(hbox_2)
        vbox.addWidget(self.rad_button_2)

        button2 = QPushButton('Browser...', self)
        button2.resize(button2.sizeHint())
        button2.clicked.connect(self.get_model_loc)
        hbox_2.addWidget(button2)

        self.text_label_2 = QLabel(self)
        self.text_label_2.setText(os.path.join(os.getcwd(), 'saved_model'))
        hbox_2.addWidget(self.text_label_2)

        grid.addWidget(group1, 0, 0)
        grid.addWidget(group2, 1, 0)
        grid.addWidget(self.pic_label, 0, 1, 2, 1)

        self.setLayout(grid)

    def get_pic_loc(self):
        path = QFileDialog.getOpenFileName(self, 'Open file', '', 'jpg Files (*.jpg)')
        if path != ('', ''):
            return path[0]

    def show_pic(self, pic):
        im = QPixmap(pic)
        self.pic_label.setPixmap(im)
        self.text_label.setText(pic)

    def load_pic(self):
        path = self.get_pic_loc()
        self.show_pic(path)

    def get_model_loc(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose catalog', '')
        if path != ('', ''):
            self.text_label_2.setText(path)
            # return path

    def set_default_model(self):
        if self.rad_button_2.isChecked():
            self.rad_button_2.setChecked(False)
            self.text_label_2.setText(os.path.join(os.getcwd(), 'saved_model'))
        else:
            self.rad_button_1.setChecked(True)

    def rad_button_2_check(self):
        if self.rad_button_1.isChecked():
            self.rad_button_1.setChecked(False)
        else:
            self.rad_button_2.setChecked(True)

    def start(self):
        self.show()


def main():
    gui = Gui()
    gui.run()


if __name__ == '__main__':
    main()
