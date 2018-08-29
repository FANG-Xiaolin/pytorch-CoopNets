from scipy.misc import imsave
import numpy as np

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding="latin1")
    fo.close()
    return dict


for j in range(1, 6):
    dataName = "data_batch_" + str(j)
    Xtr = unpickle(dataName)
    print (dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  
        img = img.transpose(1, 2, 0)  
        picName = 'cifar/' + str(i + (j - 1)*10000) + '.png'  
        imsave(picName, img)
    print (dataName + " loaded.")

print ("test_batch is loading...")


testXtr = unpickle("test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'cifar/' + str(i+50000) + '.png'
    imsave(picName, img)
print ("test_batch loaded.")
