from cProfile import run
import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2
import pathlib
from tflite_support import task

labels = ['cordon',
 'autos',
 'personas',
 'cruces',
 'pozos',
 'parar',
 'cruzar',
 'bicicleta',
 'moto',
 'escalones']

CLASSES = labels 

COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  ''' Preprocess the input image to feed to the TFLite model
  '''
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image


def detect_objects(interpreter, image, threshold):
  ''' Returns a list of detection results, each a dictionary of object info.
  '''

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  ''' Run object detection on the input image and draw the detection results
  '''
  
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(CLASSES[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8

def unique(list1):

    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list

def load_model(model_path, threshold):
  base_opt = task.core.BaseOptions(model_path)
  detection_opt = task.processor.DetectionOptions(score_threshold = threshold)
  options = task.vision.ObjectDetectorOptions(base_opt, detection_opt)
  detector = ObjectDetector.create_from_options(options)

  return detector

def detect(detector, image_path):

  classes = ['cordon',
 'autos',
 'personas',
 'cruces',
 'pozos',
 'parar',
 'cruzar',
 'bicicleta',
 'moto',
 'escalones']

  # Abro y leo la imagen como un array
  image = Image.open(image_path)
  image = np.asarray(image)

  # Divido la imagen en 3 (Izquierda, derecha y centro) utilizando el ancho de la imagen

  wid = image.shape[1]
  left = wid/3
  center = left * 2
  right = wid

  multi_detected_objects = []

  tensor_image = task.vision.TensorImage.create_from_file(image_path)
  results = detector.detect(tensor_image)
  resultsdetect = results.detections

  # Creo un randomizador de colores por clase
  CLASSES = classes
  COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

  for obj in resultsdetect:

    # Selecciono todos los ids, nombres, porcentajes de los objetos detectados

    categories = obj.categories

    for category in categories: 
      names = category.category_name
      multi_detected_objects.append(names) 
      scores = category.score * 100 
      class_id = category.index

      # Cuento la cantidad de objetos que se repiten por clase

    count_objs = {i:multi_detected_objects.count(i) for i in multi_detected_objects} 
    contador = str(count_objs)
    final_contador = contador.strip("{ }")

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
      cv2.putText(image, final_contador,(right - 200, right - 10) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Leo devuelta la imagen, para pasarla de array a imagen y luego la devuelvo
    detect_objs = unique(multi_detected_objects) 
    img = Image.fromarray(detect_img, 'RGB')
    return img, detect_objs, count_objs



#-----HEADER------

with st.container():
    st.title("IBODS Project Deployement")
    st.write("Esta es la pagina de pruebas para nuestro proyecto")

thr = st.sidebar.slider("Detection Threshold", min_value = 0.0, max_value = 1.0, value = 0.3, step = 0.01)

# model = st.sidebar.selectbox("Select Model",  ({"EfficientDet0" : 'logs/model.tflite'}, {"EfficientDet1" : ': logs/model1.tflite'}))

model_path = 'logs/model1.tflite'
threshold=0.3

detector = load_model(model_path=model_path,threshold=threshold)

image_file = st.file_uploader("Upload images for object detection", type=['png','jpeg', 'jpg'])

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image)

detect_bt = st.button("Detect objects")

if detect_bt:
    img, objs_unique, objs_contador = detect(detector,image_file)
    st.image(img)
    st.write(objs_unique)
    st.write(objs_contador)
