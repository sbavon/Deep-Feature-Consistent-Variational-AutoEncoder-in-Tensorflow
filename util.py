import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.misc
import urllib
from zipfile import ZipFile
from PIL import Image

### get the image
def crop_center_image(img):
    width_start = int(img.shape[1]/2 - 150/2)
    height_start = int(img.shape[0]/2 - 150/2)
    cropped_img = img[height_start: height_start+150, width_start: width_start+150, :]
    #print(cropped_img.shape)
    return cropped_img

### download according to address provided and perform cropping
def load_and_crop_image(img, img_width, img_height):
    img = scipy.misc.imread(img_addr)
    img = crop_center_image(img)
    img = scipy.misc.imresize(img, [img_width,img_height])
    return img

def register_extension(id, extension):
    Image.EXTENSION[extension.lower()] = id.upper()

def register_extensions(id, extensions): 
    for extension in extensions: register_extension(id, extension)

### create grid_img
### the image inputs will be 4 dimensions, which 0 dimention is the number of example
def build_grid_img(inputs, img_height, img_width, n_row, n_col):
    grid_img = np.zeros((img_height*n_row, img_width*n_col, 3))
    print(inputs.shape)
    count = 0
    for i in range(n_col):
        for j in range(n_row):
            grid_img[i*img_height:(i+1)*img_height, j*img_width:(j+1)*img_width,:] = inputs[count]
            count += 1
    return grid_img
    
### save images as a grid
def save_grid_img(inputs, path, img_height, img_width, n_row, n_col):
    
    Image.register_extension = register_extension
    Image.register_extensions = register_extensions
    grid_img = build_grid_img(inputs, img_height, img_width, n_row, n_col)
    scipy.misc.imsave(path, grid_img)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

### convert image into binary format
def get_image_binary(img):
    shape = np.array(img.shape, np.int32)
    img = np.asarray(img,np.uint8)
    return img.tobytes(), shape.tobytes()

### write data into tf record file format (images are stored in zip file)
def write_tfrecord(tfrecord_filename, zipFileName, img_height, img_width):
    
    ### images counter
    count = 0
    
    ### create a writer
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    
    with ZipFile(zipFileName) as archive:
        
        for entry in archive.infolist():
            
            # skip the folder content
            if entry.filename == 'content/':
                continue
                
            with archive.open(entry) as file:
                
                sys.stdout.write('\r'+str(count))
                
                ### pre-process data
                img = np.asarray(Image.open(file))
                img = crop_center_image(img)
                img = scipy.misc.imresize(img, [img_height,img_width])
                img, shape = get_image_binary(img)
                
                ### create features
                feature = {'image': _bytes_feature(img),
                           'shape':_bytes_feature(shape)}
                features = tf.train.Features(feature=feature)
                
                ### create example
                example = tf.train.Example(features=features)
                
                ### write example
                writer.write(example.SerializeToString())
                sys.stdout.flush()
                
                count += 1
        
        writer.close()
            
### parse serialized data back into the usable form
def _parse(serialized_data):
    features = {'image': tf.FixedLenFeature([], tf.string),
               'shape': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(serialized_data,
                                      features)
    img = tf.cast(tf.decode_raw(features['image'],tf.uint8), tf.float32)
    shape = tf.decode_raw(features['shape'],tf.int32)
    img = tf.reshape(img, shape)
    
    return img

### read tf record
def read_tfrecord(tfrecord_filename):
    
    ### create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    dataset = dataset.map(_parse)
    return dataset

def download(url, file_path):
    if os.path.exists(file_path):
        print("the file is already existed")
        return
    else:
        print("downloading file...")
    urllib.request.urlretrieve(url, file_path) 
    print("downloading done")