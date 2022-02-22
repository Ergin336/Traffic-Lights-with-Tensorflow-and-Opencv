import os
import cv2 
import glob
import applyModel
import numpy as np
from tensorflow import keras 


"""
  Este es el programa principal. En el, se encuentra la clase elaborada para la deteccion y clasificacion
  de semaforos en video e imagenes. Tambien permite obtener nuevos dataset con imagenes de semaforos
  recortadas gracias a la red neuronal entrenada con el dataset de COCO
"""


class autonomousDrive(object):
  """
    Clase para la obtencion de fijaciones de semaforos y deteccion y clasificacion de los mismos

    param: path_of_fixations: Ruta al arhivo de imagenes para recortar
    param: path_of_fixations_final: Ruta de salida para las imagenes recortadas
    param: path_of_test_images: Ruta a las imagenes de test
    param: path_of_processed_images: Ruta a las imagenes de test procesadas
    param: path_of_test_videos: Ruta a los videos de test
    param: path_of_final_videos: Ruta a los videos procesados
  """

  def __init__(self, path_of_fixations=None, path_of_fixations_final=None, path_of_test_images=None, path_of_processed_images=None, 
  path_of_test_videos=None, path_of_final_videos=None):
    self.path_of_fixations = path_of_fixations
    self.path_of_fixations_final = path_of_fixations_final
    self.path_of_test_images = path_of_test_images
    self.path_of_processed_images = path_of_processed_images
    self.path_of_test_videos = path_of_test_videos
    self.path_of_final_videos = path_of_final_videos

  def doFixations(self):
    """
      Esta funcion  se encarga de utilizar applyModel.py para extraer las fijaciones de los semaforos de nuestro dataset
      Devuelve un nuevo set de imagenes recortadas con la ROI que nos interesa
    """

    total_images = applyModel.loadImages(self.path_of_fixations + '*.jpg')
    coco_model = applyModel.downloadCocoModel() #se decarga el modelo de COCO
    total_traffic_lights = 0
    number_of_images = 0
    print("\nNumero total de imagenes:", len(total_images))
    for file in total_images:
      (processed_image_rgb, out, file_name) = applyModel.detectonImage(model=coco_model, path=file, save_fixations=None, trafficModel=None)
      if (number_of_images % 10) == 0:
        print("Total de imagenes procesadas:", number_of_images)
        print("Total de semaforos detectados: ", total_traffic_lights)

      number_of_images = number_of_images + 1

      for j in range(len(out['boxes'])): #se busca dentro de los distintos objetos detectados por COCO en una imagen
        obj_class = out["detection_classes"][j]
        if obj_class == 10: #imagenes que corresponden a semaforos
          box = out["boxes"][j]
          traffic_light = processed_image_rgb[box["y"]:box["y2"], box["x"]:box["x2"]] #se realiza el recorte de la imagen
          traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_RGB2BGR)
          cv2.imwrite(self.path_of_fixations_final + str(total_traffic_lights) + ".jpg", traffic_light)
          total_traffic_lights = total_traffic_lights + 1

    print("Numero total de semaforos detectados:", total_traffic_lights)

  def detectOnImage(self):
    """
      Funcion para la deteccion de semaforos en imagenes
    """

    files = applyModel.loadImages(self.path_of_test_images + '*.png')
    model_ssd = applyModel.downloadCocoModel()
    light_detection_nn = keras.models.load_model("drivingtrainednet.h5") #se carga el modelo entrenado de InceptionV3
    for i, file in enumerate(files):

      #se procede a la deteccion y clasificacion de semaforos
      (processed_image, out, file_name) = applyModel.detectonImage(
        model_ssd, file, save_fixations=True, trafficModel=light_detection_nn)
      cv2.imwrite(self.path_of_processed_images+ str(i) + ".jpg", cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

  def detectOnVideo(self):
    """
      Funcion para la deteccion de semaforos en video
    """

    video_shape = (1920, 1080) #resolucion para el video
    total_frames = 20.0
    model_ssd = applyModel.downloadCocoModel() #se descarga el modelo de COCO
    light_detection_nn = keras.models.load_model("drivingtrainednet.h5") #se carga el modelo entrenado de InceptionV3
    cap = cv2.VideoCapture(self.path_of_test_videos)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(self.path_of_final_videos, fourcc, total_frames, video_shape) 

    while cap.isOpened():
      success, actual_frame = cap.read() 
      if success:
        width = int(actual_frame.shape[1])
        height = int(actual_frame.shape[0])
        actual_frame = cv2.resize(actual_frame, (width, height))
        processed_frame = applyModel.detectonVideo(
          model_ssd, actual_frame, trafficModel=light_detection_nn) #se aplica el detector y clasificador a cada frame

        result.write(processed_frame)
              
      else:
        break

    cap.release()
    result.release()
    cv2.destroyAllWindows() 

  def detectManual(self, filepath, file):
    """
      Funcion sin inteligencia artificial para la deteccion de semaforos y sus colores
    """

    font = cv2.FONT_HERSHEY_SIMPLEX #fuente para color de traffic light
    img = cv2.imread(filepath+file)
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #espacio HSV

    redl = np.array([0,100,100])
    redm = np.array([10,255,255])
    redh = np.array([160,100,100])
    redsh = np.array([180,255,255])
    greenl = np.array([40,50,50])
    greenh = np.array([90,255,255])
    yellowl = np.array([15,150,150])
    yellowh = np.array([35,255,255])
    first_mask = cv2.inRange(hsv, redl, redm)
    second_mask = cv2.inRange(hsv, redh, redsh)
    third_mask = cv2.inRange(hsv, greenl, greenh)
    fourth_mask = cv2.inRange(hsv, yellowl, yellowh)
    fifth_mask = cv2.add(first_mask, second_mask)
    #máscara para cada color del semáforo a excepción del rojo donde preferimos 
    #definir una clara y oscura por variaciones en su tonalidad entre imágenes
    #de día y noche. Luego juentamientos esta ambas en una única máscara.
    shape = img.shape

    red_hough = cv2.HoughCircles(fifth_mask, cv2.HOUGH_GRADIENT, 1, 80,
                              param1=50, param2=10, minRadius=0, maxRadius=30)

    green_hough = cv2.HoughCircles(third_mask, cv2.HOUGH_GRADIENT, 1, 60,
                                param1=50, param2=10, minRadius=0, maxRadius=30)

    yellow_hough = cv2.HoughCircles(fourth_mask, cv2.HOUGH_GRADIENT, 1, 30,
                                param1=50, param2=5, minRadius=0, maxRadius=30)
    
    #Y finalmente aplicamos Hough por cada una de las máscaras de colores
    r = 5
    bound = 4.0 / 10
    if red_hough is not None:
        red_hough = np.uint16(np.around(red_hough))

        for i in red_hough[0, :]:
            if i[0] > shape[1] or i[1] > shape[0]or i[1] > shape[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= shape[0] or (i[0]+n) >= shape[1]:
                        continue
                    h += fifth_mask[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                #se insertar los círculos y texto de detección
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(third_mask, (i[0], i[1]), i[2]+30, (0, 255, 0), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1, (0,255,0), 2, cv2.LINE_AA)

    if green_hough is not None:
        green_hough = np.uint16(np.around(green_hough))

        for i in green_hough[0, :]:
            if i[0] > shape[1] or i[1] > shape[0] or i[1] > shape[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= shape[0] or (i[0]+n) >= shape[1]:
                        continue
                    h += third_mask[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                #se insertar los círculos y texto de detección
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(third_mask, (i[0], i[1]), i[2]+30, (0, 255, 0), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1, (0,255,0), 2, cv2.LINE_AA)

    if yellow_hough is not None:
        yellow_hough = np.uint16(np.around(yellow_hough))

        for i in yellow_hough[0, :]:
            if i[0] > shape[1] or i[1] > shape[0] or i[1] > shape[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= shape[0] or (i[0]+n) >= shape[1]:
                        continue
                    h += fourth_mask[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                #se insertar los círculos y texto de detección
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(third_mask, (i[0], i[1]), i[2]+30, (0, 255, 0), 2)
                cv2.putText(cimg,'Yellow',(i[0], i[1]), font, 1, (0,255,0), 2,cv2.LINE_AA)

    cv2.imwrite("/home/cesar/Downloads/trabajo_final/trabajo_final/output_images/output_classifier/01_final.jpg", cimg)
    cv2.destroyAllWindows()
            
  def HOG_SVM(self, path_train, output_img, path_model):
    """
      Función del clasificador para ampliado de zonas con más información de la imagen 
      gracias a un entrenamiento del detector SVM_HOG. Selecciona las coordenadas
      superior e inferior de entre todos los  cuadros de deteción del detector,
      que define su región de interés. 
      
      Permite mejorar los resultados de la función 'detect_manual'
    """    
    
    def getImagePaths(folder, imgExts):
      imagePaths = []
      for x in os.listdir(folder):
        xPath = os.path.join(folder, x)
        if os.path.splitext(xPath)[1] in imgExts:
          imagePaths.append(xPath)
      return imagePaths

    def getDataset(folder, classLabel):
      images = []
      labels = []
      imagePaths = getImagePaths(folder, ['.jpg', '.png', '.jpeg'])
      for imagePath in imagePaths:
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        images.append(im)
        labels.append(classLabel)
      return images, labels

    def svmInit(C, gamma):
      model = cv2.ml.SVM_create()
      model.setGamma(gamma)
      model.setC(C)
      model.setKernel(cv2.ml.SVM_LINEAR)
      model.setType(cv2.ml.SVM_C_SVC)
      model.setTermCriteria((cv2.TERM_CRITERIA_EPS + 
                             cv2.TERM_CRITERIA_MAX_ITER, 
                             1000, 1e-3))
      return model

    def svmTrain(model, samples, labels):
      model.train(samples, cv2.ml.ROW_SAMPLE, labels)

    def svmEvaluate(model, samples, labels):
      labels = labels[:, np.newaxis]
      pred = model.predict(samples)[1]
      correct = np.sum((labels == pred))
      err = (labels != pred).mean()
      print('label -- 1:{}, -1:{}'.format(np.sum(pred == 1), 
              np.sum(pred == -1)))
      return correct, err * 100

    def computeHOG(hog, images):
      hogFeatures = []
      for image in images:
        hogFeature = hog.compute(image)
        hogFeatures.append(hogFeature)
      return hogFeatures

    def prepareData(hogFeatures):
      featureVectorLength = len(hogFeatures[0])
      data = np.float32(hogFeatures).reshape(-1, featureVectorLength)
      return data

    winSize = (128, 128) # Parametros HOG para inicializar el descriptor
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = True
    nlevels = 64
    signedGradient = False

    # Inicializamos el descriptor HOG
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                          cellSize, nbins,derivAperture,
                          winSigma, histogramNormType, L2HysThreshold, 
                          gammaCorrection, nlevels,signedGradient)

    trainModel = True # Flags 
    testModel = True
    queryModel = True
    rootDir = path_train
    trainDir = os.path.join(rootDir, 'train')
    testDir = os.path.join(rootDir, 'test')

    def resize_folder(folder_path, path_result):
      """
        Función de lectura y redimensión de imágenes a una forma uniforme 
        entre todo el conjunto de entrenamiento y test. Estos nuevos 
        resultados se van a emplear en el código para entrenar el modelo.
        
        param: folder_path: ruta a carpeta con imágenes de test o entrenamiento.
        param: path_result: ruta donde genera la carpeta con los resultados del 
        redimensionamiento.
      """
      
      input_fold = folder_path
      os.mkdir(path_result)
      i = 0
      for img in sorted(glob.glob(input_fold + "/*.jpg")):
          image = cv2.imread(img)
          label = img[img.rfind("/") + 1:].split("_")[0]
          number_of_img = img.split("_")[1]
          number_of_img = str(number_of_img).replace(".jpg", "")
          img_Resized = cv2.resize(image, ((128,128)))
          cv2.imwrite(str(path_result) + ( (str(label) + '_') + "%04i.jpg" %i), img_Resized)
          i += 1
    
    # ================================ Entrenamiento del modelo =====================
    
    """
      Para cada par de carpetas de entrenamiento y test (positivo y negativo)
      aplico la función 'resize_folder' consiguiendo 'Pos_resized' y 'Neg_resized'
      de esta manera, para cada iteración de la función principal 'HOG_SVM' vamos 
      a tener que eliminar dichas carpetas localizadas en train y test dentro de
      /traffic_light_dataset.
    """
    
    if trainModel:
        print("--- Training Model ---\n")
        trainPosDir = os.path.join(trainDir, 'trainPos')
        trainNegDir = os.path.join(trainDir, 'trainNeg')
        resize_folder(trainPosDir, "./traffic_light_dataset/train/Pos_resized/")
        resize_folder(trainNegDir, "./traffic_light_dataset/train/Neg_resized/")
        trainPosDir = os.path.join(trainDir, 'Pos_resized')
        trainNegDir = os.path.join(trainDir, 'Neg_resized')
        trainPosImages, trainPosLabels = getDataset(trainPosDir, 1)
        trainNegImages, trainNegLabels = getDataset(trainNegDir, -1)
        print("Dimensiones de imágenes train positivas y negativas respectivamente")
        print(set([x.shape for x in trainPosImages]))
        print(set([x.shape for x in trainNegImages]))
        print('positivas - {}, {} || negativas - {}, {}'
              .format(len(trainPosImages),len(trainPosLabels), len(trainNegImages),len(trainNegLabels)))
        print("pos",np.array(trainPosImages).shape)
        print("neg",np.array(trainNegImages).shape)
        trainImages = np.concatenate((np.array(trainPosImages), 
                           np.array(trainNegImages)), 
                                      axis=0)
        trainLabels = np.concatenate((np.array(trainPosLabels), 
                           np.array(trainNegLabels)),
                                      axis=0)
 
        hogTrain = computeHOG(hog, trainImages)
        trainData = prepareData(hogTrain)
        print('\ntrainData: {}, trainLabels:{}'
                .format(trainData.shape, trainLabels.shape))

        model = svmInit(C=0.01, gamma=0)
        svmTrain(model, trainData, trainLabels)
        #se guardan los resultados del entrenamiento en el modelo 'traffic_lights'
        model.save(str(path_model) + 'traffic_lights.yml') 


    # ================================ Test del modelo ===============
    
    if testModel:
        print("\n\n--- Testing Model ---\n")
        model = cv2.ml.SVM_load(str(path_model) + 'traffic_lights.yml') 
        #cargo el modelo de'stop' para la detección de señales
        testPosDir = os.path.join(testDir, 'testPos')
        testNegDir = os.path.join(testDir, 'testNeg')
        resize_folder(testPosDir, "./traffic_light_dataset/test/Pos_resized/")
        resize_folder(testNegDir, "./traffic_light_dataset/test/Neg_resized/")
        testPosDir = os.path.join(testDir, 'Pos_resized')
        testNegDir = os.path.join(testDir, 'Neg_resized')
        testPosImages, testPosLabels = getDataset(testPosDir, 1)
        testNegImages, testNegLabels = getDataset(testNegDir, -1)
        print("Dimensiones de imágenes test positivas y negativas respectivamente")
        print(set([x.shape for x in testPosImages]))
        print(set([x.shape for x in testNegImages]))
        print("pos",np.array(testPosImages).shape)
        print("neg",np.array(testNegImages).shape)
        hogPosTest = computeHOG(hog, np.array(testPosImages))
        testPosData = prepareData(hogPosTest)
        posCorrect, posError = svmEvaluate(model, testPosData, 
                                           np.array(testPosLabels))
        
        tp = posCorrect
        fp = len(testPosLabels) - posCorrect
        print('\nTP: {}, FP: {}, Total: {}, error: {}'
                .format(tp, fp, len(testPosLabels), posError))

        hogNegTest = computeHOG(hog, np.array(testNegImages))
        testNegData = prepareData(hogNegTest)
        negCorrect, negError = svmEvaluate(model, testNegData, 
                                           np.array(testNegLabels))

        tn = negCorrect
        fn = len(testNegData) - negCorrect
        print('TN: {}, FN: {}, Total: {}, error: {}'
                .format(tn, fn, len(testNegLabels), negError))

        precision = tp * 100 / (tp + fp)
        recall = tp * 100 / (tp + fn)
        print('Precision: {}, Recall: {}'.format(precision, recall))


    # ================================= Consulta/Query =================================================
    
    if queryModel:
        print("\n\n--- Query of Model ---\n")
        model = cv2.ml.SVM_load(str(path_model) + 'traffic_lights.yml')
        sv = model.getSupportVectors()
        rho, aplha, svidx = model.getDecisionFunction(0)
        svmDetector = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
        svmDetector[:-1] = -sv[:]
        svmDetector[-1] = rho
        print(svmDetector.shape)
        hog.setSVMDetector(svmDetector) #se inicia el detector de semáforos
        queryImage = cv2.imread(output_img, cv2.IMREAD_COLOR) #imagen a redimensionar
        finalHeight = 800.0
        scale = finalHeight / queryImage.shape[0]
        queryImage = cv2.resize(queryImage, None, fx=scale, fy=scale)
        showed = queryImage.copy() #copiamos la imagen de entrada
        c = int(showed.shape[0]/2) #eliminamos la mitad inferior (no habrá semáforos en dicha sección)
        r = int(showed.shape[1])
        showed = showed[0:c, 0:r]
        bboxes, weights = hog.detectMultiScale(showed, winStride=(8, 8),
                                               padding=(32, 32), scale=1.05,
                                               finalThreshold=2, hitThreshold=1.0)

        max_y = showed.shape[0]
        min_y = 0
        max_x = 0
        min_x = showed.shape[1] #coordenadas de ampliación de imagen
        
        for bbox in bboxes:
          draw = False #booleando de dibujo de cuadros de detección
          x1, y1, w, h = bbox
          x2, y2 = x1 + w, y1 + h
          hsv = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
          hsv = hsv[y1:y2,x1:x2]
          if (abs( y1 - y2 )) <= 200:
              for i in hsv:
                for j in i:
                    pixel = j
                    if pixel != 0 and not draw: #hayamos píxeles no negros en HSV
                        draw = True
                        break
                  
          #se actualiza por cada cuadro en la imagen, las coordendas de la ampliación (min_x,max_y, max_x,min_y)
          if draw:
              if y1 < max_y:
                    max_y = y1

              if y2 >  min_y:
                    min_y = y2

              if x2 > max_x:
                    max_x = x2

              if x1 < min_x:
                    min_x = x1

              else:
                pass
              
          else:
            pass

        print("x1 a x2 {}-{}".format(min_x, max_x))
        print("y1 a y2 {}-{}".format(max_y, min_y)) #mostramos y aplicamos las nuevas coordenadas
        showed = showed[max_y:min_y, min_x:max_x]
        cv2.imwrite("./output_images/output_classifier/0.jpg", showed) #guarda la imagen amplida
        self.detectManual("./output_images/output_classifier/", "0.jpg")
        #se ejecuta 'detectManual' para mejorar su redimiento de detección 

  

if __name__ == "__main__":
  world_path = "D:/Universidad/Universidad-3/Vision/Practicas/trabajo_final/"
  fix_path = world_path + "JPEGImages/"
  fix_path_final = world_path + "croppedImages/"
  test_images_path = world_path + "test_images/"
  final_images_path = world_path + "output_images/output_cnn/"
  test_video_path = world_path + "test_videos/final_test2.mp4"
  final_video_path = world_path + "output_videos/final_output.mp4"
  car = autonomousDrive(fix_path, fix_path_final, test_images_path, final_images_path, test_video_path, final_video_path)
  #car.doFixations() #recortar imagenes
  #car.detectOnImage() #detecta semaforos en imagenes mediante IA
  #car.detectManual("./test_images/", "1.png") #detecta semaforos en imagenes mediante un proceso manual
  #car.detectOnVideo() #detecta semaforos en video mediante IA
  car.HOG_SVM(path_train="./traffic_light_dataset/", output_img = "./test_images/14.png", path_model="./model/")
  #ejecución de clasificador.