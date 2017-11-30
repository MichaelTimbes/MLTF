import tensorflow as tf # Prototype Tensorflow Variables
import numpy as np # Reshape Image Vectors and Matricies
from os import listdir # Directory Functions
from os import path as opath # File Path 
from skimage import io, color # Image I/O and Color Space Conversion
from PIL import Image as PImage # Image Manipulation
from PIL import ImageOps
import random
def prepareImage(fullImagePath, width, height, colormode):
    """
    Prepares an image. The image is opened, modified in memory, and returned.
    Precondition: width and height are positive
    Postcondition: none.
    Return: the modified image

    Parameters:
    fullImagePath: the full path of the image
    width: desired width of the image
    height: desired height of the image
    colormode: color mode (according to PIL documentation)
    """

    assert width > 0
    assert height > 0

    size = (width, height)
    modifiedImage = PImage.open(fullImagePath)

    modifiedImage = modifiedImage.convert(colormode)
    modifiedImage = ImageOps.fit(modifiedImage, size, PImage.BICUBIC,0,(0.5,0.5))
    return modifiedImage

def randomizeImages(sourcePath):
    """
    Reads in all files on a path, takes the ones ending with jpg, bmp, or png,
    then adds all of the non-hidden files to a list and randomizes the list.

    Precondition: none
    Postcondition: none
    Return: the randomized list of images

    Parameters:
    sourcePath: the path containing original images
    """
    images = []
    for image in listdir(sourcePath):
        if not image.startswith(".") and (image.endswith("jpeg") or image.endswith("bmp") or image.endswith("png")):
            images.append(image) 

    random.shuffle(images)
    return images

def divideImages(images, probability):
    """
    Takes a list of images and divides them into two groups: training and test.
    The number put into training is the first int(probability * len(images)) images.
    The rest go into the test set.

    Precondition: probability > 0 and probability < 1
    Postcondition: none
    Returns: two lists (trainingImages, testImages)

    Parameters:
    images: a list of image filenames
    probability: The percentage of desired training images
    """

    assert probability > 0 and probability < 1

    trainingImages = []
    testImages = []
    trainingImageCount = int(probability * len(images))
    
    i = 0
    for image in images:
        if i <= trainingImageCount:
            trainingImages.append(image)
        else:
            testImages.append(image)
        i += 1
    return (trainingImages, testImages)

def prepareAndStoreImages(images, sourcePath, savePath, width, height, colormode):
    """
    Prepares and stores image into a folder.

    Precondition: width and height are positive
    Postcondition: the images are modified and saved in the savePath
    Returns nothing.

    Parameters:
    images: a list of image filenames
    sourcePath: the source path of the original images
    savePath: the desired location of the modified images
    width: desired width of the modified images
    height: desired height of the modified images
    colormode: color mode of the modified images (according to PIL)
    """

    assert width > 0
    assert height > 0

    for image in images:
        modifiedImage = prepareImage(sourcePath + '/' + image, width, height, colormode)
        if colormode is "RGB":
            modifiedFileName = opath.splitext(image)[0] + ".jpeg"
        else:
            modifiedFileName = opath.splitext(image)[0] + ".bmp"
            
        modifiedImage.save(savePath + "/" + modifiedFileName)
        #print(image, "added to", savePath)

def sortImages(sourcePath, trainingPath, testPath, width, height, probability, colormode):
    """
    Sorts images from a source path into test and training paths.
    The number of files put into training path is determined by the probability.

    Precondition: width and height are positive, probability must be between 0 and 1.
    Postcondition: the images are modified and saved in the test and training paths
    Returns nothing.

    Parameters:
    sourcePath: the source path of the original images
    trainingPath: the desired location of the modified training images
    testPath: the desired location of the modified testing images
    width: desired width of the modified images
    height: desired height of the modified images
    probability: The percentage of desired training images
    colormode: color mode of the modified images (according to PIL)
    """

    assert width > 0
    assert height > 0
    assert probability > 0 and probability < 1
    
    images = randomizeImages(sourcePath)
    (trainingImages, testImages) = divideImages(images, probability)
    prepareAndStoreImages(trainingImages, sourcePath, trainingPath, width, height, colormode)
    prepareAndStoreImages(testImages, sourcePath, testPath, width, height, colormode)
    
def PrepImages(source_dir,train_dir,test_dir,width,height,probability,color_mode='RGB'):
    '''
    Driver for the sorting, extraction, and prep for the images needed.
    source_dir : Is the source of the images
    train_dir : The directory to save the training set
    test_dir : The directory to save the test set
    width/height : The width and height of the images
    probability : The probability of an image goining into the test set
    color_mode:  1=BW, L=Gray, RGB=Color, RGBA=Color with transparency
    '''
    probability = float(probability)
    width,height = int(width),int(height)
    
    sortImages(source_dir,train_dir,test_dir,width,height,probability,color_mode)

def labelImages(srcp,savp,label_):

    sourcePath = srcp
    savepath = savp
    nlabel = label_
    images = []

# Pull Images From Path
    for image in listdir(sourcePath):
        images.append(image)
# Rename Images
    i = 0 # Index Reference
    for image in images:
        modifiedImage = PImage.open(sourcePath + '/' + image)
        modifiedFileName = nlabel + str(i) + ".jpeg"
        i+=1
        #print(modifiedFileName)
        modifiedImage.save(sourcePath + "/" + modifiedFileName)
        
def ImportImages(path, key_dict, colortype='rgb'):
    """
    Images must be in RGB space.
    ________________________________________________________________
    Function Outline:
    1. Loads list of images from the path variable (type string).
    2. Iterates through directory loads image into I/O file stream.
    3. Converts file Numpy array.
    4. Reads labels and converts to binary class.
    5. Returns Numpy array objects for images and labels.
    """

    loadedImages = []
    loadedLabels = []
    originalLabels = []
    
    imagesList = listdir(path)
  
    for image in imagesList:
        if not(".DS_Store" in path +'/'+ image): #Issue in Mac File System
            (img, label) = loadImage(path, image, colortype)
            loadedImages.append(np.asarray( img, dtype="int32" ))
            originalLabels.append(label)
        
    # Convert to Binary Classification.
    for originalLabel in originalLabels:
        for k in key_dict.keys():
            if k in originalLabel:
                loadedLabels.append(key_dict[k])
        #if keyA in originalLabel and not(keyB in originalLabel):
         #   loadedLabels.append([1, 0])
        #else:
        #    loadedLabels.append([0, 1])

    return np.asarray(loadedImages), np.asarray(loadedLabels)

def loadImage(path, image, colortype='rgb'):
    img = PImage.open(path + '/' + image)
    label = opath.splitext(image)[:]

    # Pull file name from current image- use it as a label
    img.load()

    if colortype == 'lab':
        color.rgb2lab(img)

    # Resize step- ensures that all images follow.
    #img.thumbnail(new_size, PImage.ANTIALIAS )
    #img.convert('1')
    #img.resize(new_size)
    return (img, label[0])

def shape_up3d(data, width):
    """
    Expects a NUM_IN * NUM_IN sized picture.
    Changes the shape to be (N,NUM_IN**2).
    """
    num_exs = len(data[:,0,0,0])
    new_X = np.zeros((num_exs,width**2 * 3))
    for i in range(0,num_exs):
        new_X[i,:] = data[i,:,:,:].reshape((1,width**2 * 3))
    return new_X

def shape_up2d(data, width):
    """
    Expects a NUM_IN * NUM_IN sized picture.
    Changes the shape to be (N,NUM_IN**2).
    """
    num_exs = len(data[:,0,0])
    new_X = np.zeros((num_exs,width**2))
    for i in range(0,num_exs):
        new_X[i,:] = data[i,:,:].reshape((1,width**2))
    return new_X

def shape_up_X(data, size, num_channels,num_ex = -1):
    """
    shape_up_X(train_X, IMAG_X):
    Expects a NUM_IN * NUM_IN sized picture. Changes
    the shape to be (N,NUM_IN**2).
    ____________________________________
    """
    temp = []
    for dat in data:
        temp.append(np.reshape(dat, [1,size,size,num_channels]))
    #return np.reshape(data, [num_ex,size,size,num_channels])
    return np.asarray(temp)

def out_class(Y, keyA, keyB):
    """
    Matches Class with input vector.
    """
    return np.asarray([keyB if label == 1 else keyA for label in Y])

def weight_variable(shape):
    """
    Helper function to help initialize weight variables.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Helper function to initialize bias variable.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    Helper function that returns a 2d convolutional layer
    """
    x = x.astype(np.float32, copy=False)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')