import cv2
import os
import numpy as np

nrow=30
path='./test'
start=1
outputdir='./splited-'+path[2:]
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

def cell2img(cell_image, image_size=32, margin_syn=0):
    num_cols = nrow #cell_image.shape[1] // image_size
    num_rows = nrow #cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            images[ir*num_cols+ic] = temp
    return images

for img_name in os.listdir(path):
    print ('Spliting {}/{}'.format(start,len(os.listdir(path))))
    img=cv2.imread(os.path.join(path,img_name))
    images=cell2img(img)
    for i,splitedimg in enumerate(images):
        cv2.imwrite(os.path.join(outputdir,'{:05d}.png'.format((start-1)*nrow*nrow+i)),splitedimg)
    start+=1



