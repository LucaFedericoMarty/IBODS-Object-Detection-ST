from cProfile import run
import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2
from tflite_support.task import vision
from tflite_support.task.vision import ObjectDetector
from tflite_support.task.vision import ObjectDetectorOptions
from tflite_support.task.processor import DetectionOptions
from tflite_support.task.core import BaseOptions
from tflite_support import task

def drawBoundingBox(image_path, model_path, threshold, classes):

  # Agarro el directorio (carpeta) actual

  cwd = os.getcwd()

  # Le asigno los tres parametros:
  # - La ruta de la imagen a detectar
  # - La ruta del modelo de IA
  # - El umbral de la IA

  image_path = image_path
  model_path = model_path
  threshold = threshold

  # Leo la imagen como un tensor
  tensor_image = task.vision.TensorImage.create_from_file(image_path)

  # Abro y leo la imagen como un array
  image = Image.open(image_path)
  image = np.asarray(image)

  # Divido la imagen en 3 (Izquierda, derecha y centro) utilizando el ancho de la imagen

  wid = image.shape[1]
  left = wid/3
  center = left * 2
  right = wid 

  # Asigno las opciones de configuracion del modelo de deteccion (Que modelo, el umbral, etc)
  base_opt = BaseOptions(model_path)
  detection_opt = DetectionOptions(score_threshold = threshold)
  options = ObjectDetectorOptions(base_opt, detection_opt)

  # Creo el modelo de deteccion
  detector = ObjectDetector.create_from_options(options)

  # Agarro los resultados de la deteccion
  results = detector.detect(tensor_image)
  resultsdetect = results.detections

  # Creo un randomizador de colores por clase
  CLASSES = classes
  COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

  # Recorro cada objetos de los resultdos de deteccion

  for obj in resultsdetect:

    # Selecciono todos los ids, nombres, porcentajes de los objetos detectados

    categories = obj.categories 
    for category in categories: 
      names = category.category_name 
      scores = category.score * 100 
      class_id = category.index 

    # Creo a porcentajes en una lista

    percentages = [float(scores)]

    # Por cada id (identificador unico del objeto), le asigno un color

    color = [int(c) for c in COLORS[class_id]]

    # Usando las coordenadas del punto de abajo a la izquierda, creo el punto de arriba de la derecha mediante el width y el height, y creo la bounding box, uniendo ambos puntos

    start_point = obj.bounding_box.origin_x, obj.bounding_box.origin_y 
    end_point = obj.bounding_box.origin_x + obj.bounding_box.width, obj.bounding_box.origin_y + obj.bounding_box.height
    detect_img = cv2.rectangle(image, start_point, end_point, color, 2)

    # Creo las coordenadas de los puntos en x, los de abajo

    x1 = obj.bounding_box.origin_x
    x2 = obj.bounding_box.origin_x + obj.bounding_box.width

    # Calculo para saber la posicion del objeto
    middle = obj.bounding_box.width / 2
    Center = x2 - middle

    x = (x1, x2)

    # Comprobacion de la posicion del objeto
    if Center < left:
      position = "izquierda"
    elif Center < center:
      position = "centro"
    else:
      position = "derecha"
    
    # Creo el texto a poner y lo pongo arriba de las bounding boxes

    for percentage in percentages:
      confidence = str(int(percentage)) + '%'
      label = names + ", " + confidence
      cv2.putText(image, label, (obj.bounding_box.origin_x,obj.bounding_box.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
      cv2.putText(image, position, (obj.bounding_box.origin_x,obj.bounding_box.origin_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Leo devuelta la imagen, para pasarla de array a imagen y luego la devuelvo 
  img = Image.fromarray(detect_img, 'RGB')
  return img

#-----HEADER------

with st.container():
    st.title("IBODS Project Deployement")
    st.write("Esta es la pagina de pruebas para nuestro proyecto")

thr = st.sidebar.slider("Detection Threshold", min_value = 0.0, max_value = 1.0, value = 0.3, step = 0.01)

model = st.sidebar.selectbox("Select Model",  ({"EfficientDet0" : 'logs/model.tflite'}, {"EfficientDet1" : ': logs/model1.tflite'}))

image_file = st.file_uploader("Upload images for object detection", type=['png','jpeg'])

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image)

detect = st.button("Detect objects")

if detect:
    output_image = drawBoundingBox(input_image)
    st.image(output_image)

