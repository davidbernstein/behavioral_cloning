import csv
import cv2
import numpy as np

import os.path

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Cropping2D

from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Dropout

import sklearn
from sklearn.model_selection import train_test_split

# parameters
NUM_EPOCHS = 7
INITIAL_LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
USE_LEFT_RIGHT_CAMERAS = True
AUGMENT_WITH_FLIPPED_IMAGE = False
IMAGE_INDEX_RANGE = 3 if USE_LEFT_RIGHT_CAMERAS else 1

STEERING_CORRECTION_DEGREES = 5.0
STEERING_CORRECTION_FACTOR = STEERING_CORRECTION_DEGREES / 25.0
STEERING_CORRECTION = [0.0, STEERING_CORRECTION_FACTOR, -STEERING_CORRECTION_FACTOR]

# specify which data sets to use
DATA_SETS = []
#DATA_SETS.append('./data/udacity_data/')
DATA_SETS.append('./data/track_1_3_laps/')

DATA_SETS.append('./data/turn_1_1/')
DATA_SETS.append('./data/turn_1_2/')
DATA_SETS.append('./data/turn_1_3/')

DATA_SETS.append('./data/turn_2_1/')


def get_all_csv_lines():
	"""
	returns list of lines in all driving_log.csv files 
	included in DATA_SETS directories
	"""

	lines = []
	for data_path in DATA_SETS:
		lines += get_csv_lines(data_path)

	return lines

def get_csv_lines(data_path):
	"""
	returns list of lines in driving_log.csv files 
	in directory specified in data_path
	"""

	lines = []
	with open(data_path + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	return lines

def read_images_and_measurements():
	"""
	reads all images and steering measurements in files in 
	directories in DATA_SETS

	return value is lists of images and measurements
	"""

	images = []
	measurements = []

	for data_path in DATA_SETS:
		lines = get_csv_lines(data_path)

		for line in lines:
			for i in range(0, IMAGE_INDEX_RANGE):
				# i = [0, 1, 2] [center image, left image, right image]
				source_path = line[i]
				filename = source_path.split('/')[-1]
				current_path = data_path + 'IMG/' + filename
				image = cv2.imread(current_path)
				images.append(image)

				measurement = float(line[3]) + STEERING_CORRECTION[i]
				measurements.append(measurement)

				if AUGMENT_WITH_FLIPPED_IMAGE:
					images.append(cv2.flip(image,1))
					measurements.append(-measurement)

	return images, measurements

def generator(samples, batch_size=32):
	"""
	image reading generator, mainly copied from the Udacity video lecture
	modified to look for images in multiple directories

	samples: list of lines in driving_log.csv files
	"""

    num_samples = len(samples)

    # Loop forever so the generator never terminates
    while True: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
            	for i in range(0, IMAGE_INDEX_RANGE):
	            	source_path = batch_sample[i]
	            	filename = source_path.split('/')[-1]

	            	# look in paths in DATA_SETS to find image file
	            	image = None
	            	for data_path in DATA_SETS:
	            		current_path = data_path + 'IMG/' + filename
	            		if os.path.exists(current_path):
	            			image = cv2.imread(current_path)

	            	# FIXME: raise exception here
	            	if image == None:
	            		print('image not found')

	            	measurement = float(batch_sample[3]) + STEERING_CORRECTION[i]

	            	images.append(image)
	            	measurements.append(measurement)

	            	if AUGMENT_WITH_FLIPPED_IMAGE:
	            		images.append(cv2.flip(image,1))
	            		measurements.append(-measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def LeNet():
	"""
	LeNet NN architecture
	"""

	model = Sequential()

	# add lambda layers for image preprocessing
	# image normalization and mean centering
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

	# image cropping
	model.add(Cropping2D(cropping=((70,25), (0,0))))

	model.add(Convolution2D(20, 5, 5, border_mode='same'))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Convolution2D(50, 5, 5, border_mode='same'))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))

	return model

def Nvidia():
	"""
	Nvidia NN architecture
	modified to include a Dropout layer
	"""

	model = Sequential()

	# add lambda layers for image preprocessing
	# image normalization and mean centering
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

	# image cropping
	model.add(Cropping2D(cropping=((70,25), (0,0))))

	# convolution layers
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'relu'))

	model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))

	model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'relu'))

	model.add(Convolution2D(64,3,3, activation = 'relu'))

	model.add(Convolution2D(64,3,3, activation = 'relu'))

	# experimental dropout layer
	model.add(Dropout(0.1))

	# dense layers
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Dense(128))
	model.add(Dense(32))

	# output layer
	model.add(Dense(1))

	return model

def main():
	"""
	specify, train, and save model
	"""

	# define model
	#model = LeNet()
	model = Nvidia()

	# using generator
	if True:
		lines = get_all_csv_lines()
		train_samples, validation_samples = train_test_split(lines, test_size=VALIDATION_SPLIT)
		
		training_samples_per_epoch = len(train_samples)
		validation_samples_per_epoch = len(validation_samples)

		if USE_LEFT_RIGHT_CAMERAS:
			training_samples_per_epoch *= 3
			validation_samples_per_epoch *= 3

		if AUGMENT_WITH_FLIPPED_IMAGE:
			training_samples_per_epoch *= 2
			validation_samples_per_epoch *= 2

		train_generator = generator(train_samples, batch_size=32)
		validation_generator = generator(validation_samples, batch_size=32)

		model.compile(loss='mse', optimizer='adam', lr = INITIAL_LEARNING_RATE)
		model.fit_generator(train_generator, 
							samples_per_epoch=training_samples_per_epoch, 
							validation_data=validation_generator, 
	            			nb_val_samples=validation_samples_per_epoch, 
	            			nb_epoch=NUM_EPOCHS)
	else:
		# without generator
		images, measurements = read_images_and_measurements()

		X_train = np.array(images)
		y_train = np.array(measurements)

		model.compile(loss='mse', optimizer='adam')
		model.fit(X_train, y_train, 
				  validation_split=VALIDATION_SPLIT, 
				  shuffle=True, 
				  nb_epoch=NUM_EPOCHS)


	# save model to .h5 file
	model.save('model.h5')

if __name__ == "__main__":
	main()
