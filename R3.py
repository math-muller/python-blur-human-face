import cv2
import numpy as np
 
path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
path2 = cv2.data.haarcascades + "haarcascade_eye.xml"

# Inicializa o classificador cascade
face_classifier = cv2.CascadeClassifier(path) 

# configura a captura de imagem da webcam
video_cap = cv2.VideoCapture(0)

# se a webcam abrir pego um frame
if video_cap.isOpened():
    rval, frame = video_cap.read()
    
else:
    rval = False

while rval:
    # Exibe saida da imagem
    cv2.imshow("normal", frame)
    cv2.imshow("result", frame)

    # Atualiza com um novo frame
    rval, frame = video_cap.read()

    # ESC para sair do programa
    key = cv2.waitKey(10)
    if key == 27:
        break

# Função de callback, quando ocorre um evento do mouse, essa função é chamada
def mouse_click(event, x, y, flags, param):
    global frame
    
    # Se foi o botão esquerdo do mouse  
    if event == cv2.EVENT_LBUTTONDOWN:
        # Converte o frame para escala de cinza 
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Realiza a detecção de face na imagem em cinza
        faces_return = face_classifier.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
        # Faz a varredura na lista de faces detectadas em faces_return
        for (x,y,w,h) in faces_return:
            # Desenha um retangulo em cada face detectada
            
            img = frame.copy()

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)

            # Aplica uma mascara no frame completo
            img[y:y+h, x:x+w] = cv2.medianBlur(img[y:y+h, x:x+w],35)

            # Exibe saida da imagem
            cv2.imshow("result", img)
        
    # Se foi o botão direito do mouse  
    if event == cv2.EVENT_RBUTTONDOWN:
        print('test')
    



# Seta a função de callback que será chamada 
# Evento 'image', função callback mouse_click  
cv2.setMouseCallback("result", mouse_click)
cv2.waitKey(1)
# fecha a janela.
video_cap.release()
cv2.destroyAllWindows()