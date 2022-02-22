import cv2 
import glob 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
 

"""
  Este programa es uno de los principales del trabajo, ya que permite obtener las fijaciones de semaforos
  a partir de imagenes normales utilizando una red neuronal entrenada con el dataset de COCO.
  Tambien realiza la clasificacion de los semaforos tanto en imagenes como en video
  gracias a la red neuronal InceptionV3, que se entrena en trainNeuralNet.py sumada a la otra red entrenada con COCO
"""


def boxCenter(box, coordinates):
  """
    Funcion para generar el centro de la caja delimitadora

    param: box: Caja delimitadora en cuestion
    param: coordinates: Coordendas de la caja bajo x e y
  """
  return (box[coordinates] + box[coordinates + "2"]) / 2
 
def threshBox(boxes, box_index, thresh):
  """
    Funcion que elimina las cajas delimitadoras duplicadas

    param: boxes: Cajas delimitadoras
    param: box_index: Lugar de la caja en cuestion
    param: thresh: Umbral para determinar cuando es aceptable una caja
  """

  box = boxes[box_index]
 
  for idx in range(box_index):
    other_box = boxes[idx]
    if abs(boxCenter(other_box, "x") - boxCenter(box, "x")) < thresh and abs(boxCenter(other_box, "y") - boxCenter(box, "y")) < thresh:
      return False
 
  return True
 
def loadImages(path):
  """
    Funcion que genera una lista para cargar las imagenes de un directorio

    param: path: Ruta al directiorio
  """

  images = []
  for filename in glob.iglob(path, recursive=True):
    images.append(filename)
 
  return images

def mixImages(images, labels):
  """
    Funcion que mezcla las imagenes para añadir una componente de azar

    param: images: Lista de imagenes a mezclar
    param: labels: Etiquetas de las imagenes (green, red, yellow, other_object)
  """
  indexes = np.random.permutation(len(images))
 
  return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]
 
def reverseModel(detectedImage):
  """
    Funcion que invierte el proceso de preprocesado
    Esta funcion se utiliza dentro del entrenamiento de la red
    para conseguir invertir el procesado anterior

    param: detectedImage: Imagen que ha sido procesada
  """

  image = detectedImage + 1.0
  image = image * 127.5
  return image.astype(np.uint8)
     
def downloadModel(model_object_name):
  """
    Funcion para descargar el modelo de red neuronal entrenado bajo el dataset de COCO
    
    param: Nombre del modelo preentrenado
  """

  url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + model_object_name + '.tar.gz'
     
  whereisModel = tf.keras.utils.get_file(fname=model_object_name, untar=True, origin=url) #es necesario cargarlo en la memoria cache
  whereisModel = str(whereisModel) + "/saved_model"
  model = tf.saved_model.load(str(whereisModel))
 
  return model

def downloadCocoModel():
  """
    Funcion que aporta el nombre del modelo entrenado a descargar
  """
  
  return downloadModel("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8")
 
def other2Rgb(path, shape=None):
  """
    Funcion que convierte las imagenes al espacio de color RGB
     
    param: path: Ruta a las imagenes
    param: shape: Dimensiones de la imagen
  """

  images = loadImages(path)
  images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in images]
 
  if shape:
    return [cv2.resize(image, shape) for image in images]
  
  else:
    return images
 
def getFix(naturalImage, path, last_image, trafficModel=None):
  """
    Funcion auxiliar de detectonImage(). Esta funcion permite, dentro de las salidas de la 
    red entrenada por COCO, detectar exclusivamente los semaforos y, en caso de tener 
    la variable trafficModel, clasificarlos segun su color mediante la red neuronal InceptionV3

    param: naturalImage: Imagen en el espacio RGB
    param: path: Ruta al archivo de imagenes
    param: last_image: Imagen despues de haber sido procesada por la red neuronal entrenada con COCO
    param: trafficModel: Parametro utilizado para realizar 
    una deteccion en profundidad del semaforo (no solamente lo detecta, 
    sino que al tener este parametro, tambien clasifica su color)
  """

  last_image_file = path.replace('.jpg', '_test.jpg')
  for idx in range(len(last_image['boxes'])): #dentro de cada caja devuelta por el modelo de COCO se extraen los objetos 
    image_type = last_image["detection_classes"][idx]
    score = int(last_image["detection_scores"][idx] * 100) 
    box = last_image["boxes"][idx] #caja delimitadora
 
    color = None
    light_descriptor = ""

    #se realiza la deteccion de un semaforo
    if image_type == 10:
      light_descriptor = "Traffic Light " + str(score)
      
      #si hay un modelo de InceptionV3, se clasifica el semaforo detectado
      if trafficModel:
        image_traffic_light = naturalImage[box["y"]:box["y2"], box["x"]:box["x2"]]
        image_inception = cv2.resize(image_traffic_light, (299, 299))
        image_inception = np.array([preprocess_input(image_inception)])
        prediction = trafficModel.predict(image_inception) #se aplica el modelo InceptionV3
        label = np.argmax(prediction)
        score_light = str(int(np.max(prediction) * 100))

        #en las siguientes lineas se delcaran los colores de clasificacion
        if label == 0:
          light_descriptor = "Green " + score_light
          color = (0, 255, 0)
       
        elif label == 1:
          light_descriptor = "Yellow " + score_light
          color = (255, 255, 0)
       
        elif label == 2:
          light_descriptor = "Red " + score_light
          color = (255, 0, 0)
        
        else: #para no saturar de informacion la imagen, se decide no etiquetar los elementos que no son semaforo
          light_descriptor = str(0)

    #se dibujan los elementos de clasificacion en torno al semaforo
    if color and light_descriptor and threshBox(last_image["boxes"], idx, 5.0) and score > 50:
      cv2.rectangle(naturalImage, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
      cv2.putText(naturalImage, light_descriptor, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
  #estas dos lineas de codigo no son extrictamente necesarias ya que en el programa de autonomDrive
  #se guardan las imagenes resultantes en un directorio especifico
  cv2.imwrite(last_image_file, cv2.cvtColor(naturalImage, cv2.COLOR_RGB2BGR))
  print(last_image_file)
 
 
def detectonImage(model, path, save_fixations=False, trafficModel=None):
  """
    Una de las funciones principales de este programa. Se encarga de aplicar el modelo entrenado
    con el dataset de COCO a las imagenes que se le pasen como parametro.
    Con la salida del modelo, se decide si se quieren extraer todos los
    objetos clasificados por la red neuronal en una imagen (parametro save_fixations=False
    para el posterior recorte de las imagenes de semaforos) o si se quiere aplicar el modelo
    entrenado de COCO + el modelo InceptionV3 que se entrena en el programa trainNeuralNet 
    (parametro save_fixations=True para clasificar si un semaforo esta en verde, rojo o amarillo)
    para realizar la correcta deteccion y clasificacion de semaforos en una imagen

    param: model: Modelo entrenado de la red neuronal con COCO
    param: path: Ruta a la imagen
    param: save_fixations: Parametro que permite aplicar la red neuronal InceptionV3 (True)
    param: trafficModel: Parametro utilizado en la funcion auxiliar getFix() para realizar 
    una deteccion en profundidad del semaforo (no solamente lo detecta, 
    sino que al tener este parametro, tambien clasifica su color)
  """

  image_bgr = cv2.imread(path)
  naturalImage = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  input_tensor = tf.convert_to_tensor(naturalImage) # Input needs to be a tensor
  input_tensor = input_tensor[tf.newaxis, ...]
 
  #se le pasa una imagen de entrada a la red entrenada por COCO
  last_image = model(input_tensor)
  num_detections = int(last_image.pop('num_detections'))
  last_image = {key: value[0, :num_detections].numpy()
            for key, value in last_image.items()}
  last_image['num_detections'] = num_detections
  last_image['detection_classes'] = last_image['detection_classes'].astype(np.int64)
  last_image['boxes'] = [
    {"y": int(box[0] * naturalImage.shape[0]), "x": int(box[1] * naturalImage.shape[1]), "y2": int(box[2] * naturalImage.shape[0]),
     "x2": int(box[3] * naturalImage.shape[1])} for box in last_image['detection_boxes']]
 
  #se decide si se quieren extraer los semaforos o simplemente devolver la salida del modelo de red de COCO
  if save_fixations:
    getFix(naturalImage, path, last_image, trafficModel)
 
  return naturalImage, last_image, path
     
def detectonVideo(model, processFrames, trafficModel=None):
  """
    Una de las funciones principales de este programa. Se encarga de aplicar 
    el modelo entrenado con el dataset de COCO al frame del video que se 
    le pase como parametro. Se compone de aplicar en primer lugar la red entrenada 
    con el dataset de COCO para detectar multiples objetos y, finalmente, a esas 
    detecciones se le aplica la red neuronal entrenada en trainNeuralNet InceptionV3 
    para detectar y clasificar semaforos
    
    param: model: Modelo entrenado de la red neuronal con COCO
    param: processFrames: Frame actual del video a procesar
    param: trafficModel: Parametro utilizado para realizar 
    una deteccion en profundidad del semaforo (no solamente lo detecta, 
    sino que al tener este parametro, tambien clasifica su color)

  """
  
  naturalImage = cv2.cvtColor(processFrames, cv2.COLOR_BGR2RGB)
  input_tensor = tf.convert_to_tensor(naturalImage) 
  input_tensor = input_tensor[tf.newaxis, ...]

  #se le pasa una imagen de entrada a la red entrenada por COCO
  last_image = model(input_tensor)
  num_detections = int(last_image.pop('num_detections'))
  last_image = {key: value[0, :num_detections].numpy()
            for key, value in last_image.items()}
  last_image['num_detections'] = num_detections
  last_image['detection_classes'] = last_image['detection_classes'].astype(np.int64)
  last_image['boxes'] = [
    {"y": int(box[0] * naturalImage.shape[0]), "x": int(box[1] * naturalImage.shape[1]), "y2": int(box[2] * naturalImage.shape[0]),
     "x2": int(box[3] * naturalImage.shape[1])} for box in last_image['detection_boxes']]
 

  for idx in range(len(last_image['boxes'])): #dentro de cada caja devuelta por el modelo de COCO se extraen los objetos
    image_type = last_image["detection_classes"][idx] 
    score = int(last_image["detection_scores"][idx] * 100)
    box = last_image["boxes"][idx] #caja delimitadora
    color = None
    light_descriptor = ""

    #se realiza la deteccion de un semaforo
    if image_type == 10:
      light_descriptor = "Traffic Light " + str(score)
      
      #si hay un modelo de InceptionV3, se clasifica el semaforo detectado
      if trafficModel:
        image_traffic_light = naturalImage[box["y"]:box["y2"], box["x"]:box["x2"]]
        image_inception = cv2.resize(image_traffic_light, (299, 299))
        image_inception = np.array([preprocess_input(image_inception)])
        prediction = trafficModel.predict(image_inception) #se aplica el modelo InceptionV3
        label = np.argmax(prediction)
        score_light = str(int(np.max(prediction) * 100))

        #en las siguientes lineas se delcaran los colores de clasificacion
        if label == 0:
          light_descriptor = "Green " + score_light
          color = (0, 255, 0)
        
        elif label == 1:
          light_descriptor = "Yellow " + score_light
          color = (255, 255, 0)
        
        elif label == 2:
          light_descriptor = "Red " + score_light
          color = (255, 0, 0)
        
        else: #para no saturar de informacion la imagen, se decide no etiquetar los elementos que no son semaforo
          light_descriptor = str(0)
 
    #se dibujan los elementos de clasificacion en torno al semaforo
    if color and light_descriptor and threshBox(last_image["boxes"], idx, 5.0) and score > 20:
      cv2.rectangle(naturalImage, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
      cv2.putText(naturalImage, light_descriptor, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
  last_frame = cv2.cvtColor(naturalImage, cv2.COLOR_RGB2BGR)
  return last_frame