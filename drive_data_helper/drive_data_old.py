import numpy as np
import cv2
import csv
import sklearn
from sklearn.model_selection import train_test_split


def _get_drive_logs(data_path):
    '''
    _get_drive_logs
    Reads the driving_log.csv file and returns a list of log strings
    :param data_path the path where the data files are located
    :return list[String] - list of driving log entries
    '''
    drive_logs = []
    with open(data_path + '/driving_log.csv') as f:
        has_header = sniff_header(f)
        reader = csv.reader(f)
        if has_header:
            print('skipping header...')
            next(reader, None)  # skip header

        for line in reader:
            drive_logs.append(line)

    print('Total drive logs:', len(drive_logs))
    return drive_logs

def sniff_header(f) :
    '''
    samples a few lines from the input files and check to see if it has a header
    the method also rewinds the file pointer back to 0 so the actual file processing
    can work from the beginning.
    :param f: file pointer
    :return: boolean : has_header
    '''
    sample=''
    for i in range(0,5) :
        sample += f.readline()
    has_header = csv.Sniffer().has_header(sample)
    f.seek(0)
    return has_header

def load_image(imageUrl, data_path):
    '''
    load_image
    parses out the image filename from the url and then loads it from the
    relative data_path
    :param imageUrl : the imageURL from the driving log
    :param data_path: the data path where the images and driving logs are stored
    :return: image as numpy array (float32)
    '''

    filename = imageUrl.split('/')[-1]
    image = cv2.cvtColor(cv2.imread(data_path + '/IMG/' + filename), cv2.COLOR_BGR2RGB)
    return np.array(image, dtype=np.float32)


def _load_image_and_labels(drive_logs, data_path, correction_factor):
    '''
    load image and labels
    loads the center, left and right images and their correspongind steering angles
    for each drive log. The images are returned as features and the steering anlges are
    the regression labels.

    The steering angles for center image is provided in the driving log. For left and right
    steering angles, we adjust the center steering angle with the correction factor (add for
    left and subtract for right)

    :param drive_logs: list of drive log string array. The first 3 array elements contain
    the center, left and right image urls. Strings 4 to 7 represent steering angle, throttle,
    break and speed.
    :param data_path: path to the data folder
    :param correction_factor: factor to adjust the left and right steering angles
    :return:
    '''
    images = []
    steering_angles = []
    for drive_log in drive_logs:
        # center image
        center_image = load_image(drive_log[0], data_path)
        steering_angle = float(drive_log[3])

        images.append(center_image)
        steering_angles.append(steering_angle)

        # left image
        # left_image = load_image(drive_log[1], data_path)
        # images.append(left_image)
        # steering_angles.append(steering_angle + correction_factor)

        # right image
        # right_image = load_image(drive_log[2], data_path)
        # images.append(right_image)
        # steering_angles.append(steering_angle - correction_factor)

    return (np.array(images), np.array(steering_angles))


def load(data_path, correction_factor=0.2):
    '''
    load -  helper method that will load the drive logs and creates the
    training dataset with images as features and steering_angle as labels

    NOTE: This method will try to load all images and labels in memory and may not
    scale for large training size sets. use the load_generators method instead

    :param : data_path : the relative path where the data files (images and driving
    logs) are stored
    :return : (X,y) : X numpy array with dimension m*4 containing m RGB images and
                y numpy array(m*1) dimension containing the steering angles
    '''

    drive_logs = _get_drive_logs(data_path)
    (X, y) = _load_image_and_labels(drive_logs, data_path, correction_factor)
    return (X, y)


def generator(drive_logs, data_path, correction_factor, batch_size=32):
    '''
    generator

    returns a generator object that would generate the image features and steering angle
    labels

    for every call to the generator, batch_size of the drive logs are processed. For each
    we load the center, left and right images and their correspongind steering angles.
    The images are returned as features and the steering anlges are the regression labels.

    The steering angles for center image is provided in the driving log. For left and right
    steering angles, we adjust the center steering angle with the correction factor (add for
    left and subtract for right)

    :param drive_logs: list of drive log string array. The first 3 array elements contain
    the center, left and right image urls. Strings 4 to 7 represent steering angle, throttle,
    break and speed.
    :param data_path: path to the data folder
    :param correction_factor: factor to adjust the left and right steering angles
    :param batch_size: factor to adjust the left and right steering angles
    :return:
    '''
    num_logs = len(drive_logs)
    while 1:  # to keep the generator running for multiple epochs

        sklearn.utils.shuffle(drive_logs)

        for offset in range(0, num_logs, batch_size):

            batch_logs = drive_logs[offset:offset + batch_size]
            images = []
            steering_angles = []

            for drive_log in batch_logs:
                # center image
                center_image = load_image(drive_log[0], data_path)
                center_steering = float(drive_log[3])

                images.append(center_image)
                steering_angles.append(center_steering)

                # Augment images with flipped and other mirror views

                # center_flipped
                add_flipped(center_image, center_steering, images, steering_angles)

                # left image
                left_image = load_image(drive_log[1], data_path)
                images.append(left_image)
                left_steering = center_steering + correction_factor
                steering_angles.append(left_steering)

                # left _flipped
                add_flipped(left_image, left_steering, images, steering_angles)

                # right image
                right_image = load_image(drive_log[2], data_path)
                images.append(right_image)
                right_steering = center_steering - correction_factor
                steering_angles.append(right_steering)

                # right_flipped
                add_flipped(right_image, right_steering, images, steering_angles)

            X_train = np.array(images)
            y_train = np.array(steering_angles)

            yield sklearn.utils.shuffle(X_train, y_train)


def add_flipped(image, steering, images, steering_angles):
    flipped_image = cv2.flip(image, 1)
    flipped_steering = steering * -1
    images.append(flipped_image)
    steering_angles.append(flipped_steering)


def load_generators(data_path, correction_factor=0.2):
    '''
    load - the main helper method that will load the drive logs and creates the
    training dataset with images as features and steering_angle as labels
    :param : data_path : the relative path where the data files (images and driving
    logs) are stored
    :return : tuple (train_generator, total_train_size, valid_generator, total_valid_size)
    '''

    drive_logs = _get_drive_logs(data_path)
    train_logs, valid_logs = train_test_split(drive_logs, test_size=0.2)

    train_generator = generator(train_logs, data_path, correction_factor, batch_size=100)
    valid_generator = generator(valid_logs, data_path, correction_factor, batch_size=100)

    total_train_size = len(train_logs) * 6  # because we are adding 3 images (c, l, r) per drivelog & 3 flipped for each
    total_valid_size = len(valid_logs) * 6

    return (train_generator, total_train_size, valid_generator, total_valid_size)
