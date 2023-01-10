import tensorflow as tf

### NNCLR Hyperparameters
try:
    AUTOTUNE = tf.data.AUTOTUNE     
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

# The values below should be changed to the appropriate values for your dataset.
unlabelled_instances = 380
labelled_train_instances = 329

temperature = 0.1        # the temperature for the softmax function in the contrastive loss
queue_size = 1000        # the size of the queue for storing the feature vectors

input_shape = (96, 1)       # the input shape of each sequence
width = 64                  # the size of the output embedding vector for each sequence
pretrain_num_epochs = 500   # the number of epochs to pretrain the model
finetune_num_epochs = 100    # The number of epochs to fine-tune the model.
BATCH_SIZE = 16             # the batch size for training
SHUFFLE_BUFFER_SIZE = 1000  # the buffer size for shuffling the dataset
k_size = 16                 # the size of the kernel for the 1D convolutional layer in the encoder
n_classes = 6               # the number of classes in the dataset

t_names = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']