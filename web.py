#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import multiprocessing as mp

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from werkzeug import secure_filename
import os
import eval

app = Flask(__name__)
app.config['DEBUG'] = True
UPLOAD_FOLDER = './static/images/tmp'

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/post', methods=['GET','POST'])
def post():
  if request.method == 'POST':
    if not request.files['file'].filename == u'':
      f = request.files['file']
      img_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
      f.save(img_path)
      result = eval.evaluation(img_path, './model.ckpt')
    else:
      result = []
    return render_template('index.html', result=result)
  else:
    return redirect(url_for('index'))

if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0')