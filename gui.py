from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton, QGridLayout, QGroupBox, \
    QHBoxLayout, QVBoxLayout, QRadioButton
from PyQt6.QtGui import QPixmap
import os, sys


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

        self.text_label = QLabel(self)
        hbox.addWidget(self.text_label)

        self.pic_label = QLabel(self)

        button1 = QPushButton('Browser...', self)
        button1.resize(button1.sizeHint())
        button1.clicked.connect(self.load_pic)
        hbox.addWidget(button1)

        rad_button_1 = QRadioButton('Default')
        rad_button_1.setChecked(True)
        vbox.addWidget(rad_button_1)

        rad_button_2 = QRadioButton('Choose my own:')
        vbox.addWidget(rad_button_2)

        grid.addWidget(group1, 0, 0)
        grid.addWidget(group2, 1, 0)
        grid.addWidget(self.pic_label, 0, 1, 2, 1)

        self.setLayout(grid)

    def get_loc(self):
        path = QFileDialog.getOpenFileName(self, 'Open file', '', 'jpg Files (*.jpg)')
        if path != ('', ''):
            return path[0]

    def show_pic(self, pic):
        im = QPixmap(pic)
        self.pic_label.setPixmap(im)
        self.text_label.setText(pic)

    def load_pic(self):
        path = self.get_loc()
        self.show_pic(path)

    def start(self):
        self.show()


if __name__ == '__main__':
    gui = Gui()
    gui.run()
