# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import heapq

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  # assert(type(images) == list)
  # assert(type(images[0]) == np.ndarray)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  # inps = []
  # for img in images:
  # img = img.astype(np.float32)
  #  inps.append(np.expand_dims(img, 0))
  bs = 100
  with tf.Session() as sess:
    writer = tf.summary.FileWriter('summary/test', sess.graph)
    
    preds = []
    n_batches = int(math.ceil(float(len(images)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = images[(i * bs):min((i + 1) * bs, len(images))]
        # inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def get_top5error(images, category):
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 100
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('summary/test', sess.graph)

        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        err = 0;
        for tt in range(preds.shape[0]):
            pred  = preds[tt, :]
            idx = heapq.nlargest(5, range(len(pred)), pred.take)
            if (category in idx) == False:
                err = err + 1;
        return err/preds.shape[0]

def get_average_probability_test(images):
  assert (type(images) == list)
  assert (type(images[0]) == np.ndarray)
  assert (len(images[0].shape) == 3)
  assert (np.max(images[0]) > 10)
  assert (np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
  bs = 100
  with tf.Session() as sess:
      writer = tf.summary.FileWriter('summary/test', sess.graph)

      preds = []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      for i in range(n_batches):
          sys.stdout.write(".")
          sys.stdout.flush()
          inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
          inp = np.concatenate(inp, 0)
          pred = sess.run(softmax, {'ExpandDims:0': inp})
          preds.append(pred)
      preds = np.concatenate(preds, 0)
      a = preds[1,:]
      return heapq.nlargest(3, range(len(a)), a.take), heapq.nlargest(3, a)

def get_average_probability(images, category):
  assert (type(images) == list)
  assert (type(images[0]) == np.ndarray)
  assert (len(images[0].shape) == 3)
  assert (np.max(images[0]) > 10)
  assert (np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
  bs = 100
  with tf.Session() as sess:
      writer = tf.summary.FileWriter('summary/test', sess.graph)

      preds = []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      for i in range(n_batches):
          sys.stdout.write(".")
          sys.stdout.flush()
          inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
          inp = np.concatenate(inp, 0)
          pred = sess.run(softmax, {'ExpandDims:0': inp})
          preds.append(pred)
      preds = np.concatenate(preds, 0)
      return np.mean(preds[:, category])



# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)  #use this line if you are using lower version of tensorflow
            # o.set_shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w) #use this line if you are using lower version of tensorflow
    # logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()


def cell2img(cell_image, image_size=32, margin_syn=0):
    num_cols = cell_image.shape[1] // image_size
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            images[ir*num_cols+ic] = temp
    return images


def main():
    dir = './test-inception'
    from model.utils.data_io import DataSet
    imglist=DataSet(dir,image_size=32)
    img=imglist.data()
    imgd=((img+1.0)/2.0*255)
    imglist=imgd
    m,s=get_inception_score(imglist)
    print ("=== Size: {}, mean {}, sd {} ===".format(len(imglist), m, s))

if __name__ == '__main__':
    main()



