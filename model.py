import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model


def get_main_path() -> str:
    main_folder = os.getcwd()
    main_folder = main_folder + '\\Rice_Image_Dataset'
    return main_folder


def display_multiple_samples(rows: int, cols: int, main_folder: str) -> None:
    types_of_rice = list(os.listdir(main_folder))
    types_of_rice.remove('Rice_Citation_Request.txt')
    for rice in types_of_rice:
        rice_folder = rice
        if rice == 'Basmati':
            rice = 'basmati'
        figure, ax = plt.subplots(rows, cols, figsize=(20, 10))
        plt.suptitle(rice.upper())
        for n in range(cols*rows):
            rice_image_name = f'{rice} ({int(np.random.randint(10000, 10100, 1))}).jpg'
            rice_image_path = os.path.join(main_folder, rice_folder, rice_image_name)
            image = plt.imread(rice_image_path)
            ax.ravel()[n].imshow(image)
            ax.ravel()[n].set_axis_off()
        plt.tight_layout()
        plt.show()


"""
def log_rice_types(name, table_name, main_folder: str):
    # run = wandb.init(project='rice_types_classification', job_type='image_visualization', name=name, anonymous='allow')
    image_table = wandb.Table(columns=['Rice Type', 'Image1', 'Image2', 'Image3', 'Image4', 'Image5'])
    types_of_rice = list(os.listdir(main_folder))
    types_of_rice.remove('Rice_Citation_Request.txt')
    for rice in types_of_rice:
        rice_folder = rice
        if rice == 'Basmati':
            rice = 'basmati'
        image_names = [f'{rice} ({int(np.random.randint(10000, 10100, 1))}).jpg' for _ in range(5)]
        paths = [os.path.join(main_folder, rice_folder, image_name) for image_name in image_names]
        image_table.add_data(rice.upper(), wandb.Image(paths[0]), wandb.Image(paths[1]), wandb.Image(paths[2]),
                             wandb.Image(paths[3]), wandb.Image(paths[4]))
    # wandb.log({table_name: image_table})
    wandb.finish()
"""


def distribution_of_rice_data(main_folder: str) -> None:
    types_of_rice = list(os.listdir(main_folder))
    types_of_rice.remove('Rice_Citation_Request.txt')
    num_images = [len(os.listdir(os.path.join(main_folder, rice))) for rice in types_of_rice]
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    plt.suptitle('Distribution of Rice Types')
    ax1.bar(types_of_rice, num_images)
    ax1.set_xlabel('Type')
    ax1.set_ylabel('Number')
    plt.tight_layout()
    plt.show()


def data_augmentation(main_folder: str) -> pd.DataFrame:
    names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    data_frames = []
    for index, name in enumerate(names):
        # na potrzeby testów tylko 5000 z każdego typu, bo inaczego długo zajmuje
        data_frames.append(pd.DataFrame({'filepath': [os.path.join(main_folder, name,
                                                      os.listdir(os.path.join(main_folder, name))[i])
                                         for i in tqdm(range(1000), position=0, leave=True)], 'label': index + 1}))
    df = pd.concat(data_frames, axis=0)
    df['label'] = df['label'].astype(str)
    return df


def generate_image_data(df: pd.DataFrame) -> tuple:
    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=df,
                                                  x_col='filepath',
                                                  y_col='label',
                                                  subset='training',
                                                  batch_size=32,
                                                  shuffle=True,
                                                  class_mode='categorical',
                                                  target_size=(224, 224))

    test_generator = datagen.flow_from_dataframe(dataframe=df,
                                                 x_col="filepath",
                                                 y_col='label',
                                                 batch_size=32,
                                                 subset='validation',
                                                 shuffle=False,
                                                 class_mode='categorical',
                                                 target_size=(224, 224))
    return train_generator, test_generator


def visualize_augmented_image(to_visualize) -> None:
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


def create_model(traing_generator, test_generator, file_name):
    vgg = VGG16(input_shape=[224, 224, 3], weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    x = Flatten()(vgg.output)
    prediction = Dense(units=5, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.compile(loss='BinaryCrossentropy' , optimizer='adam', metrics=['accuracy'])
    # aktualnie 1 poch, zeby sprawdzic zapisywanie, potem zwiekszyc ilosc do 5
    model.fit_generator(traing_generator, epochs=1, validation_data=test_generator)
    model.save(file_name)


def main():
    main_folder = get_main_path()
    # display some samples
    # display_multiple_samples(5, 5, main_folder)
    # display distribution of data
    # distribution_of_rice_data(main_folder)
    df = data_augmentation(main_folder)
    df = shuffle(df)
    df.reset_index(drop=True, inplace=True)
    file_name = 'saved_model'
    train, test = generate_image_data(df)
    create_model(train, test, file_name)
    # article base: https://www.kaggle.com/code/padmanabhanporaiyar/rice-type-classification-99-accuracy-w-b
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/ zapis modelu
    # (na później), dopytać które lepsze
    # jakie gui? (pomysł: wybieramy ścieżkę do pliku (windows eksplorator plików), lub zestawu i wyświetlamy zdjęcia z
    # odpowiednimi opisami
    # czy coś jeszcze brakuje koncepcyjnie?


if __name__ == '__main__':
    main()
