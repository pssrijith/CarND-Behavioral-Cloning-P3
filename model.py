import sys
from drive_data_helper import drive_data
from net_arch import lenet_arch, nvid_arch

data_path = './dataset'
if len(sys.argv) > 1 :
    data_path = sys.argv[1]

# load train , valid and test input from the drive data logs
# use drive_data helper methods to load data
(train_input, valid_input, test_input) = drive_data.load_input(data_path)

# for each input log we have 6 images (center, left, right and their mirror images)
train_size = len(train_input) * 6
valid_size = len(valid_input) * 6
test_size = len(test_input) * 6

print("train size={} ,valid size={}, test_size={}".format(train_size,valid_size,test_size ))

# create train and valid generators
correction_factor=0.2
batch_size = 100
train_generator = drive_data.get_generator(train_input, data_path, correction_factor, batch_size)
valid_generator = drive_data.get_generator(valid_input, data_path, correction_factor, batch_size)

# build the conv net arch with Keras
model = nvid_arch.build_nvidia()
model.compile(loss='mse', optimizer='adam')

history_obj = model.fit_generator(train_generator,samples_per_epoch=train_size,
                    validation_data=valid_generator, nb_val_samples=valid_size,
                    nb_epoch=10, verbose =1)

# print the training loss and validation loss for each epoch
print("training_loss_hist")
print(history_obj.history['loss'])
print("valid_loss_hist")
print(history_obj.history['val_loss'])

# get the test score
print("Test metrics")
test_generator = drive_data.get_generator(test_input, data_path,correction_factor, batch_size = 100)
test_loss = model.evaluate_generator(test_generator,test_size)
print("test_metrics : ", test_loss)

# save the model
model.save('model.h5')
