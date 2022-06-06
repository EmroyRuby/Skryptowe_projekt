import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model, load_model


class MyModel:
    def __init__(self, use_default=True, file_name='saved_model', *, activation='softmax', loss='BinaryCrossentropy',
                 optimizer='adam', metrics=['accuracy'], epochs=5):
        self.main_folder = os.getcwd() + '\\Rice_Image_Dataset'
        if use_default:
            if os.path.exists(os.path.join(os.getcwd(), 'saved_model')):
                self.model = load_model(os.path.join(os.getcwd(), 'saved_model'))
            else:
                self.create_default_model()
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
                                             for i in tqdm(range(1000), position=0, leave=True)], 'label': index + 1}))
        self.df = pd.concat(data_frames, axis=0)
        self.df['label'] = self.df['label'].astype(str)

    def generate_image_data(self) -> tuple:
        self.df = shuffle(self.df)
        datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, validation_split=0.2)
        train_generator = datagen.flow_from_dataframe(dataframe=self.df,
                                                      x_col='filepath',
                                                      y_col='label',
                                                      subset='training',
                                                      batch_size=32,
                                                      shuffle=True,
                                                      class_mode='categorical',
                                                      target_size=(224, 224))

        test_generator = datagen.flow_from_dataframe(dataframe=self.df,
                                                     x_col="filepath",
                                                     y_col='label',
                                                     batch_size=32,
                                                     subset='validation',
                                                     shuffle=False,
                                                     class_mode='categorical',
                                                     target_size=(224, 224))
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
        vgg = VGG16(input_shape=[224, 224, 3], weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False
        x = Flatten()(vgg.output)
        prediction = Dense(units=5, activation='softmax')(x)
        model = Model(inputs=vgg.input, outputs=prediction)
        model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit_generator(traing_generator, epochs=5, validation_data=test_generator)
        model.save(file_name)

    def create_default_model(self):
        self.data_augmentation()
        train, test = self.generate_image_data()
        self.create_model(train, test)

    def train_and_save_special_model(self, file_name, activation, loss, optimizer, metrics, epochs):
        self.data_augmentation()
        traing_generator, test_generator = self.generate_image_data()
        vgg = VGG16(input_shape=[224, 224, 3], weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False
        x = Flatten()(vgg.output)
        prediction = Dense(units=5, activation=activation)(x)
        model = Model(inputs=vgg.input, outputs=prediction)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit_generator(traing_generator, epochs=epochs, validation_data=test_generator)
        model.save(file_name)


def main():
    model = MyModel()


if __name__ == '__main__':
    main()
