from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton, QGridLayout, QGroupBox, \
    QHBoxLayout, QVBoxLayout, QRadioButton, QMenu, QLineEdit
from PyQt6.QtGui import QPixmap, QFont, QMovie
from model import MyModel
import exception
import os
import sys
import time


class Gui(QApplication):
    def __init__(self):
        super().__init__([])
        self.window = Window()

    def run(self):
        self.window.start()
        sys.exit(self.exec())



class LoadingWindow(QWidget):
    def __init__(self):
        super().__init__()

        grid = QGridLayout()

        label = QLabel(self)
        label.setText('Your model is now being created,\nthis process can take a while to complete')

        grid.addWidget(label, 0, 0)
        self.setLayout(grid)

    def start(self):
        self.show()

    def stop(self):
        self.close()


class ErrorWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Error')
        self.setGeometry(400, 400, 200, 50)

        grid = QGridLayout()

        self.error_msg = QLabel(self)
        self.error_msg.setFont(QFont('Arial', 15))
        self.error_msg.setText('Please fill all parameters!')

        button = QPushButton('Ok', self)
        button.clicked.connect(lambda: self.close())

        grid.addWidget(self.error_msg, 0, 0)
        grid.addWidget(button)
        self.setLayout(grid)

    def pass_exception(self, text):
        self.error_msg.setText(text)


class SecondWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.error_msg = ErrorWindow()
        self.loading_window = LoadingWindow()

        self.setWindowTitle('Model generator')
        self.setGeometry(300, 300, 400, 200)

        grid = QGridLayout()

        label_1 = QLabel(self)
        label_1.setText('model name:')

        self.text_input = QLineEdit(self)

        label_2 = QLabel(self)
        label_2.setText('activation type:')

        self.popup_button_1 = QPushButton('activation', self)
        menu_1 = QMenu(self)
        menu_1.addAction('relu')
        menu_1.addAction('sigmoid')
        menu_1.addAction('softmax')
        menu_1.addAction('selu')
        menu_1.addAction('elu')
        menu_1.triggered.connect(lambda x: self.popup_button_1.setText(x.text()))
        self.popup_button_1.setMenu(menu_1)

        label_3 = QLabel(self)
        label_3.setText('loss type:')

        self.popup_button_2 = QPushButton('loss', self)
        menu_2 = QMenu(self)
        menu_2.addAction('BinaryCrossentropy')
        menu_2.addAction('CategoricalCrossentropy')
        menu_2.addAction('Poisson')
        menu_2.addAction('SparseCategoricalCrossentropy')
        menu_2.triggered.connect(lambda x: self.popup_button_2.setText(x.text()))
        self.popup_button_2.setMenu(menu_2)

        label_4 = QLabel(self)
        label_4.setText('optimizer type:')

        self.popup_button_3 = QPushButton('optimizer', self)
        menu_3 = QMenu(self)
        menu_3.addAction('SGD')
        menu_3.addAction('RMSprop')
        menu_3.addAction('Adam')
        menu_3.addAction('Adadelta')
        menu_3.addAction('Adamax')
        menu_3.triggered.connect(lambda x: self.popup_button_3.setText(x.text()))
        self.popup_button_3.setMenu(menu_3)

        label_5 = QLabel(self)
        label_5.setText('metrics type:')

        self.popup_button_4 = QPushButton('metrics', self)
        menu_4 = QMenu(self)
        menu_4.addAction('accuracy')
        menu_4.addAction('binary_accuracy')
        menu_4.addAction('binary_crossentropy')
        menu_4.addAction('mean_squared_error')
        menu_4.addAction('hinge')
        menu_4.triggered.connect(lambda x: self.popup_button_4.setText(x.text()))
        self.popup_button_4.setMenu(menu_4)

        label_6 = QLabel(self)
        label_6.setText('number of epochs:')

        self.text_input_2 = QLineEdit(self)

        empty_label = QLabel(self)

        button_1 = QPushButton('Generate', self)
        button_1.clicked.connect(self.generate_button)

        button_2 = QPushButton('Cancel', self)
        button_2.clicked.connect(lambda: self.close())

        grid.addWidget(label_1, 0, 0)
        grid.addWidget(self.text_input, 0, 1)
        grid.addWidget(label_2, 1, 0)
        grid.addWidget(self.popup_button_1, 1, 1)
        grid.addWidget(label_3, 2, 0)
        grid.addWidget(self.popup_button_2, 2, 1)
        grid.addWidget(label_4, 3, 0)
        grid.addWidget(self.popup_button_3, 3, 1)
        grid.addWidget(label_5, 4, 0)
        grid.addWidget(self.popup_button_4, 4, 1)
        grid.addWidget(label_6, 5, 0)
        grid.addWidget(self.text_input_2, 5, 1)
        grid.addWidget(empty_label, 6, 0, 1, 2)
        grid.addWidget(button_2, 7, 0)
        grid.addWidget(button_1, 7, 1)
        self.setLayout(grid)

    def generate_button(self):
        try:
            file_name = self.text_input.text()
            if file_name == '':
                raise exception.MissingFileName
            activation = self.popup_button_1.text()
            if activation == 'activation':
                raise exception.MissingActivationArg
            loss = self.popup_button_2.text()
            if loss == 'loss':
                raise exception.MissingLossArg
            optimizer = self.popup_button_3.text()
            if optimizer == 'optimizer':
                raise exception.MissingOptimizerArg
            metrics = self.popup_button_4.text()
            if metrics == 'metrics':
                raise exception.MissingMetricsArg
            if not self.text_input_2.text().isdigit():
                raise exception.EpochsNotANumber
            epochs = int(self.text_input_2.text())
            self.loading_window.start()
            self.create_model_with_param(file_name, activation, loss, optimizer, metrics, epochs)
            self.loading_window.stop()
        except exception.MissingFileName:
            self.error_msg.pass_exception('File name is missing')
            self.error_msg.show()
        except exception.MissingActivationArg:
            self.error_msg.pass_exception('Activation argument is missing')
            self.error_msg.show()
        except exception.MissingLossArg:
            self.error_msg.pass_exception('Loss argument is missing')
            self.error_msg.show()
        except exception.MissingOptimizerArg:
            self.error_msg.pass_exception('Optimizer argument is missing')
            self.error_msg.show()
        except exception.MissingMetricsArg:
            self.error_msg.pass_exception('Metrics argument is missing')
            self.error_msg.show()
        except exception.EpochsNotANumber:
            self.error_msg.pass_exception('Number of epochs is not an integer')
            self.error_msg.show()

    def create_model_with_param(self, file_name, activation, loss, optimizer, metrics, epochs):
        metrics = [metrics]
        model = MyModel(use_default=False, use_existing=False, file_name=file_name, activation=activation, loss=loss,
                        optimizer=optimizer, metrics=metrics, epochs=epochs)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.another_window = SecondWindow()

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

        button3 = QPushButton('Run', self)
        button3.resize(button3.sizeHint())
        button3.clicked.connect(self.run_button)

        button4 = QPushButton('Create my own model', self)
        button4.resize(button4.sizeHint())
        button4.clicked.connect(self.create_my_model)

        self.text_label_3 = QLabel(self)

        grid.addWidget(group1, 0, 0)
        grid.addWidget(group2, 1, 0)
        grid.addWidget(self.pic_label, 0, 1, 2, 1)
        grid.addWidget(button3, 2, 0)
        grid.addWidget(self.text_label_3, 2, 1)
        grid.addWidget(button4, 3, 0)

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

    def run_button(self):
        model = MyModel(use_default=False, use_existing=True, path_of_model=self.text_label_2.text())
        path_to_img = self.text_label.text()
        pred = model.use_model(path_to_img)
        pred = pred[0]
        types = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        max_value = pred[0]
        max_index = 0
        for index, val in enumerate(pred):
            if val > max_value:
                max_value = val
                max_index = index
        self.text_label_3.setText(f'Type: {types[max_index]}')

    def create_my_model(self):
        self.another_window.show()

    def start(self):
        self.show()


def main():
    gui = Gui()
    gui.run()


if __name__ == '__main__':
    main()
