import cv2
import numpy as np
 
path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
path2 = cv2.data.haarcascades + "haarcascade_eye.xml"

# Inicializa o classificador cascade
face_classifier = cv2.CascadeClassifier(path) 
olhos_classifier = cv2.CascadeClassifier(path2) 

# configura a captura de imagem da webcam
video_cap = cv2.VideoCapture(0)

# se a webcam abrir pego um frame
if video_cap.isOpened():
    rval, frame = video_cap.read()
    
else:
    rval = False


# Função de callback, quando ocorre um evento do mouse, essa função é chamada
def mouse_click(event, x, y, flags, param):


    # Se foi o botão esquerdo do mouse  
    if event == cv2.EVENT_LBUTTONDOWN:
        pass
    # Se foi o botão direito do mouse  
    if event == cv2.EVENT_RBUTTONDOWN:
        pass


# Seta a função de callback que será chamada 
# Evento 'image', função callback mouse_click  
cv2.setMouseCallback('image', mouse_click)
   
cv2.waitKey(0)
  
# fecha a janela.
cv2.destroyAllWindows()