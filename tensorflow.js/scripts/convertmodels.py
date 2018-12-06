# import keras
# import tensorflowjs as tfjs
# from keras.models import load_model

# models = ['01_noise.h5', '02_noise.h5', '03_noise.h5', '04_noise.h5']

# for m in models:
#   model = load_model(m)
#   tfjs.converters.save_keras_model(model, 'web/'+m.split('_')[0])
#   print 'converted', m


import keras
import tensorflowjs as tfjs
from keras.models import load_model

models = ['models/VGG19_length_variation_2/01_noise.h5']

for m in models:
  model = load_model(m)
  tfjs.converters.save_keras_model(model, 'web/length/'+m.split('_')[0])
  print 'converted', m

