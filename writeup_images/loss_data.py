import matplotlib.pyplot as plt


#training_loss = [0.0182, 0.0086, 0.0065, 0.0055, 0.0051, 0.0045, 0.0041]
#validation_loss = [0.0101, 0.0075, 0.0060, 0.0056, 0.0052, 0.0046, 0.0052]

training_loss = [0.0141, 0.0076, 0.0058, 0.0048, 0.0042, 0.0036, 0.0032]
validation_loss = [0.0082, 0.0063, 0.0049, 0.0047, 0.0048, 0.0046, 0.0041]

# 1 0.0182 0.0101
# 2 0.0086 0.0075
# 3 0.0065 0.0060
# 4 0.0055 0.0056
# 5 0.0051 0.0052
# 6 0.0045 0.0046
# 7 0.0041 0.0052

plt.plot(training_loss)
plt.plot(validation_loss)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()