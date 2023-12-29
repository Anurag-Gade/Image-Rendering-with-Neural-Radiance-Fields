from utils import *
from encoder_params import *
from train_params import *

import_libraries()


def display(model, image_size):
    X,Y =np.meshgrid(np.linspace(0,image_size,image_size),np.linspace(0,image_size,image_size))
    indexes = np.concatenate([Y.reshape([image_size*image_size,1]),X.reshape([image_size*image_size,1])],axis=1)
    y =model(torch.from_numpy(indexes.astype(np.float32)/image_size))
    image_g =np.reshape(y.detach().numpy(),[image_size,image_size,3])
    
    img =cv2.imread('test.jpg')
    img2 =cv2.imread('test2.jpg')

    plt.figure(figsize=(16,3))
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title('Original Image.')
    plt.subplot(1,4,2)
    plt.imshow(image_g.astype(np.uint8))
    plt.title('MLP Image.')
    plt.show()
