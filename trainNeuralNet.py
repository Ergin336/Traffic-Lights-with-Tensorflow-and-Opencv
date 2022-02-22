import cv2 
import sys
import applyModel 
import collections
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
sys.path.append('../')


"""
  Este programa es el encargado de entrenar y guardar los parametros de la red neuronal InceptionV3. 
  Esta red es la que permite detectar y clasificar semaforos correctamente en una imagen una vez haya sido 
  procesada por la otra red entrenada en COCO (es necesario extraer los objectos detectados en un primer lugar).
  En este proyecto, se utilizan cuatro conjuntos de datos:

  -Imagenes de semaforo en verde recortadas 
  -Imagenes de semaforo en rojo recortadas
  -Imagenes de semaforo en amarillo recortadas
  -Images de cualquier cosa que no sea un semaforo recortadas o no recortadas 
"""


def showTrained(trained):
  """
    Funcion para visualizar el acierto de la red durante su entrenamiento

    param: trained: Se trata de los valores de coste y metricas durante los diferentes epochs o etapas
  """
  
  plt.plot(trained.history['accuracy'])
  plt.plot(trained.history['val_accuracy'])
  plt.title('Modelo de exactitud')
  plt.ylabel('Exactitud')
  plt.xlabel('Epoch')
  plt.legend(['Exactitud de entrenamiento', 'Exactitud de validacion'], loc='best')
  plt.show()
 
def recursiveLearning(classSize, stopLearning=True):
  """
    Funcion que entrena la red neuronal InceptionV3 mediante transferencia
    Esta funcion devuelve la red lo mejor entrenada posible (pasando un umbral minimo)
     
    param: classSize: Numero total de clases
    param: stopLearning: Le comunica a la red si debe cambiar sus parametros o no
  """

  """
    La arquitectura de InceptionV3 se inicializa con los siguientes parametros:

    -weights: Se escoge un pre-entrenamiento con ImageNet
  
    -include_top: En este caso se elimina la parte alta de la arquitectura, la cual corresponde a un clasificador
                  ya preparado
  
    -input_shape: Necesita tener al menos los tres canales para poder realizar el procesamiento de imagenes 
                  correctamente
  """

  main_net = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
  print("InceptionV3 cargada con exito!\n")
  print('Capas de la red: ', len(main_net.layers))
  print("Dimensiones de las distintas salidas:", main_net.outputs)
  main_net.summary()

  """
    Se ha eliminado el clasificador predefinido de InceptionV3 para poder adaptar el nuestro propio:

    -Sequential: Hemos decidio utilizar este modelo ya que implica un tensor de entrada y otro de salida.
                 Resulta obvio pensar que para el caso de procesamiento de imagenes (que en este codigo
                 se tratan como tensores) es un modelo muy adecuado para conformar su clasificacion
   
    -GlobalAveragePooling2D: Agrupacion promedio para los datos espaciales
    
    -Dropout: Establecemos un dropout que no de riesgo a obtener overfitting
   
    -Dense: Utilizamos Dense para efectuar correctamente la funcion de activacion a la red neuronal
            Es importante destacar que hacemos uso tanto de la funcion relu como softmax para 
            obtener una mayor eficiencia computacional y mejores resultados. Relu nos permite procesar
            a una gran velocidad ya que aporta suficiente linealidad, ademas, es raro que sature.
            Sin embargo, softmax es necesaria ya que relu tiende a poner sus gradientes a 0 en el proceso 
            de backpropagation cuando cualquiera de sus neuronas tambien sea 0
  """

  first_part_model = Sequential()
  first_part_model.add(main_net)
  first_part_model.add(GlobalAveragePooling2D())
  first_part_model.add(Dropout(0.5))
  first_part_model.add(Dense(1024, activation='relu'))
  first_part_model.add(BatchNormalization())
  first_part_model.add(Dropout(0.5))
  first_part_model.add(Dense(512, activation='relu'))
  first_part_model.add(Dropout(0.5))
  first_part_model.add(Dense(128, activation='relu'))
  first_part_model.add(Dense(classSize, activation='softmax'))

  #se detiene la actualizacion de parametros dentro la red cuando no se quiere que se siga entrenando
  if stopLearning:
    for layer in main_net.layers:
      layer.trainable = False
 
  return first_part_model

#se generan distintos lotes de tensores con aumento de datos en tiempo real
conform_data = ImageDataGenerator(rotation_range=5, width_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             zoom_range=[0.7, 1.5], height_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             horizontal_flip=True)
 
dim = (299, 299)
 
#en este caso estos son los arhivos del dataset (rojo, verde, amarillo, ninguno)
red_path = "D:/Universidad/Universidad-3/Vision/Practicas/Trabajo_fin_de_materia/trabajo_final/traffic_light_dataset/2_red/*"
green_path = "D:/Universidad/Universidad-3/Vision/Practicas/Trabajo_fin_de_materia/trabajo_final/traffic_light_dataset/0_green/*"
yellow_path = "D:/Universidad/Universidad-3/Vision/Practicas/Trabajo_fin_de_materia/trabajo_final/traffic_light_dataset/1_yellow/*"
other_object_path = "D:/Universidad/Universidad-3/Vision/Practicas/Trabajo_fin_de_materia/trabajo_final/traffic_light_dataset/3_not/*"

#todas las imagenes se convierten al espacio RGB
img_red = applyModel.other2Rgb(red_path, dim)
img_green = applyModel.other2Rgb(green_path, dim)
img_yellow = applyModel.other2Rgb(yellow_path, dim)
img_other_stuff = applyModel.other2Rgb(other_object_path, dim)
labels = [0] * len(img_green)
labels.extend([1] * len(img_yellow))
labels.extend([2] * len(img_red))
labels.extend([3] * len(img_other_stuff))

fix = np.ndarray(shape=(len(labels), 4))
array_of_images = np.ndarray(shape=(len(labels), dim[0], dim[1], 3))
dataset_images = []
dataset_images.extend(img_green)
dataset_images.extend(img_yellow)
dataset_images.extend(img_red)
dataset_images.extend(img_other_stuff)

assert len(dataset_images) == len(labels)  
dataset_images = [preprocess_input(picture) for picture in dataset_images]
(dataset_images, labels) = applyModel.mixImages(dataset_images, labels)
 
for idx in range(len(labels)):
  array_of_images[idx] = dataset_images[idx]
  fix[idx] = labels[idx]
     
print("\nTotal de imagenes a procesar: ", len(dataset_images))
print("Numero total de etiquetas: ", len(labels))

for idx in range(len(fix)):
  fix[idx] = np.array(to_categorical(labels[idx], 4))

train_info = int(len(fix) * 0.8)
x_train = array_of_images[0:train_info]
x_valid = array_of_images[train_info:]
y_train = fix[0:train_info]
y_valid = fix[train_info:]
 
number_of_lights = collections.Counter(labels)
print('\nEtiquetas durante el entrenamiento:', number_of_lights)
n = len(labels)
print('0:', number_of_lights[0])
print('1:', number_of_lights[1])
print('2:', number_of_lights[2])
print('3:', number_of_lights[3])
 
ponderate_traffic_light = {0: n / number_of_lights[0], 1: n / number_of_lights[1], 2: n / number_of_lights[2], 3: n / number_of_lights[3]}
print('Peso por clase:', ponderate_traffic_light)

#se entrena y guarda el modelo de forma recursiva
save_model = ModelCheckpoint("drivingtrainednet.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
stop_training = EarlyStopping(min_delta=0.0005, patience=15, verbose=1)
model = recursiveLearning(classSize=4, stopLearning=True)
model.summary()

it_train = conform_data.flow(x_train, y_train, batch_size=32)
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(
  learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
 
#se utilizan 250 etapas para el entrenamiento aunque normalmente da buenos resultados a partir de las 50
trained_object = model.fit(it_train, epochs=250, validation_data=(
  x_valid, y_valid), shuffle=True, callbacks=[
  save_model, stop_training], class_weight=ponderate_traffic_light)
showTrained(trained_object)
 
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Perdida en la validacion:', score[0])
print('Acierto en la validacion:', score[1])

for idx in range(len(x_valid)):
  img_as_ar = np.array([x_valid[idx]]) 
  prediction = model.predict(img_as_ar)
  label = np.argmax(prediction)
  file_name = str(idx) + "_" + str(label) + "_" + str(np.argmax(str(y_valid[idx]))) + ".jpg"
  img = img_as_ar[0]

  #se invierte el procesado
  img = applyModel.reverseModel(img)
  cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))