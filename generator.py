from utils import *
from encoder_params import *
from train_params import *

import_libraries()


def generator(batch_size):

    image = cv2.imread('test.jpg')
    X = []
    y = []

    list_to_shuffle = [] #We used 100, as that is the side of the square image
    for k in range(100):
      list_to_shuffle.append(k)
    random.shuffle(list_to_shuffle)
    while True:
        for i in list_to_shuffle:
            for j in list_to_shuffle:
                X.append([i,j])
                y.append(image[i,j])
                if len(X) == batch_size:
                    yield (np.array(X,dtype = np.float32)/100,np.array(y,dtype = np.float32))
                    X=[]
                    y=[]
     