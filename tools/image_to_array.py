from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os


def plotImage(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_list = os.path.join('C:/Study/RAP-course-work/RAP_dataset/')

    BATCH_SIZE = 10
    IMG_SHAPE = 100

    train_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_list,
                                                               shuffle=True,
                                                               target_size=(IMG_SHAPE, IMG_SHAPE),
                                                               class_mode='binary')

    sample_training_images, _ = next(train_data_gen)
    plotImage(sample_training_images)