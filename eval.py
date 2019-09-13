#!/usr/bin/env python
#! -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import tensorflow as tf
import os
import random
import main

cascade_path = './opencv/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

HUMAN_NAMES = {
  0: u"ジョン・レノン",
  1: u"ジョン・レノン",
  2: u"ポール・マッカトニー",
  3: u"ポール・マッカトニー",
  4: u"ジョージ・ハリスン",
  5: u"ジョージ・ハリスン",
  6: u"リンゴ・スター",
  7: u"リンゴ・スター"
}

def evaluation(img_path, ckpt_path):
  tf.reset_default_graph()
  f = open(img_path, 'r')
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = faceCascade.detectMultiScale(gray, 1.1, 3)
  if len(face) > 0:
    for rect in face:
      random_str = str(random.random())
      cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=2)
      face_detect_img_path = './static/images/tmp/' + random_str + 'a.jpg'
      cv2.imwrite(face_detect_img_path, img)
      x = rect[0]
      y = rect[1]
      w = rect[2]
      h = rect[3]
      cv2.imwrite('./static/images/tmp/' + random_str + 'b.jpg', img[y:y+h, x:x+w])
      target_image_path = './static/images/tmp/' + random_str + 'b.jpg'
  else:
    print ('image:NoFace')
    return
  f.close()

  f = open(target_image_path, 'r')
  image = []
  img = cv2.imread(target_image_path)
  img = cv2.resize(img, (28, 28))
  image.append(img.flatten().astype(np.float32)/255.0)
  image = np.asarray(image)
  logits = main.inference(image, 1.0)
  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  sess.run(tf.initialize_all_variables())
  if ckpt_path:
    saver.restore(sess, ckpt_path)

  softmax = logits.eval()
  result = softmax[0]
  rates = [round(n * 100.0, 1) for n in result]
  humans = []

  humans.append({
      'label': 0,
      'name': HUMAN_NAMES[0],
      'rate': round(rates[0] + rates[1], 2)
    })
  humans.append({
      'label': 1,
      'name': HUMAN_NAMES[2],
      'rate': round(rates[2] + rates[3], 2)
    })
  humans.append({
      'label': 2,
      'name': HUMAN_NAMES[4],
      'rate': round(rates[4] + rates[5], 2)
    })
  humans.append({
      'label': 3,
      'name': HUMAN_NAMES[6],
      'rate': round(rates[6] + rates[7], 2)
    })

  rank = sorted(humans, key=lambda x: x['rate'], reverse=True)
  return [rank, face_detect_img_path, target_image_path]

if __name__ == '__main__':
  evaluation('testimage.jpg', './model.ckpt')