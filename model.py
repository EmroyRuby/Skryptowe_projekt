import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.models import Model, load_model, Sequential


class MyModel:
    def __init__(self, use_default=True, use_existing=False, path_of_model='', *, file_name='saved_model',
                 activation='softmax', loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'], epochs=5):
        self.main_folder = os.getcwd() + '\\Rice_Image_Dataset'
        if use_default:
            if os.path.exists(os.path.join(os.getcwd(), 'saved_model')):
                self.model = load_model(os.path.join(os.getcwd(), 'saved_model'))
            else:
                self.create_default_model()
        elif use_existing:
            if os.path.exists(path_of_model):
                self.model = load_model(path_of_model)
            else:
                print('No model')
        else:
            self.train_and_save_special_model(file_name, activation, loss, optimizer, metrics, epochs)

    def display_multiple_samples(self, rows: int, cols: int) -> None:
        types_of_rice = list(os.listdir(self.main_folder))
        types_of_rice.remove('Rice_Citation_Request.txt')
        for rice in types_of_rice:
            rice_folder = rice
            if rice == 'Basmati':
                rice = 'basmati'
            figure, ax = plt.subplots(rows, cols, figsize=(20, 10))
            plt.suptitle(rice.upper())
            for n in range(cols*rows):
                rice_image_name = f'{rice} ({int(np.random.randint(10000, 10100, 1))}).jpg'
                rice_image_path = os.path.join(self.main_folder, rice_folder, rice_image_name)
                image = plt.imread(rice_image_path)
                ax.ravel()[n].imshow(image)
                ax.ravel()[n].set_axis_off()
            plt.tight_layout()
            plt.show()

    def show_distribution_of_rice_data(self) -> None:
        types_of_rice = list(os.listdir(self.main_folder))
        types_of_rice.remove('Rice_Citation_Request.txt')
        num_images = [len(os.listdir(os.path.join(self.main_folder, rice))) for rice in types_of_rice]
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        plt.suptitle('Distribution of Rice Types')
        ax1.bar(types_of_rice, num_images)
        ax1.set_xlabel('Type')
        ax1.set_ylabel('Number')
        plt.tight_layout()
        plt.show()

    def data_augmentation(self) -> None:
        names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        data_frames = []
        for index, name in enumerate(names):
            # na potrzeby testów tylko 5000 z każdego typu, bo inaczego długo zajmuje
            data_frames.append(pd.DataFrame({'filepath': [os.path.join(self.main_folder, name,
                                                          os.listdir(os.path.join(self.main_folder, name))[i])
                                             for i in tqdm(range(14000), position=0, leave=True)], 'label': index + 1}))
        self.df = pd.concat(data_frames, axis=0)
        self.df['label'] = self.df['label'].astype(str)

    def generate_image_data(self) -> tuple:
        self.df = shuffle(self.df)
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = datagen.flow_from_dataframe(dataframe=self.df,
                                                      x_col='filepath',
                                                      y_col='label',
                                                      subset='training',
                                                      batch_size=32,
                                                      shuffle=True,
                                                      class_mode='categorical',
                                                      target_size=(32, 32))

        test_generator = datagen.flow_from_dataframe(dataframe=self.df,
                                                     x_col="filepath",
                                                     y_col='label',
                                                     batch_size=32,
                                                     subset='validation',
                                                     shuffle=False,
                                                     class_mode='categorical',
                                                     target_size=(32, 32))
        return train_generator, test_generator

    def visualize_augmented_image(self, to_visualize) -> None:
        for batch in to_visualize:
            images = batch[0]
            labels = batch[1]
            for i in range(5):
                plt.figure(figsize=(20, 10))
                plt.imshow(images[i])
                print(images[i].shape)
                plt.show()
                print(labels[i])
            break

    def create_model(self, traing_generator, test_generator, file_name='saved_model'):
        self.model = Sequential()
        self.model.add(Conv2D(16, 3, activation='relu'))
        self.model.add(Conv2D(16, 3, activation='relu'))
        self.model.add(MaxPool2D())
        self.model.add(Conv2D(32, 3, activation='relu'))
        self.model.add(Conv2D(32, 3, activation='relu'))
        self.model.add(MaxPool2D())
        self.model.add(Flatten())
        self.model.add(Dense(5, 'softmax'))
        self.model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit_generator(traing_generator, epochs=20, validation_data=test_generator)
        self.model.save(file_name)

    def create_default_model(self):
        self.data_augmentation()
        train, test = self.generate_image_data()
        self.create_model(train, test)

    def train_and_save_special_model(self, file_name, activation, loss, optimizer, metrics, epochs):
        self.data_augmentation()
        traing_generator, test_generator = self.generate_image_data()
        self.model = Sequential()
        self.model.add(Conv2D(16, 3, activation=activation))
        self.model.add(Conv2D(16, 3, activation=activation))
        self.model.add(MaxPool2D())
        self.model.add(Conv2D(32, 3, activation=activation))
        self.model.add(Conv2D(32, 3, activation=activation))
        self.model.add(MaxPool2D())
        self.model.add(Flatten())
        self.model.add(Dense(5, 'softmax'))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        self.model.fit_generator(traing_generator, epochs=epochs, validation_data=test_generator)
        self.model.save(file_name)

    def use_model(self, path):
        image = load_img(path, target_size=(32, 32))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        return self.model.predict(image)

    def train(self):
        self.data_augmentation()
        traing_generator, test_generator = self.generate_image_data()
        self.model.fit_generator(traing_generator,
                                 steps_per_epoch=traing_generator.samples//traing_generator.batch_size,
                                 epochs=5, validation_data=test_generator,
                                 validation_steps=test_generator.samples//test_generator.batch_size, verbose=1)
        self.model.save('saved_model')

def main():
    model = MyModel()
    result = model.use_model(os.path.join(os.getcwd(), 'Rice_Image_Dataset', 'Basmati', 'basmati (2).jpg'))
    print(result)


if __name__ == '__main__':
    main()
