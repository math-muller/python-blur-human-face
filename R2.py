import cv2
import numpy as np
import imutils

path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
path2 = cv2.data.haarcascades + "haarcascade_eye.xml"

hat = cv2.imread("hat2.png", cv2.IMREAD_UNCHANGED)

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

while rval:
    # Converte o frame para escala de cinza 
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Realiza a detecção de face na imagem em cinza
    faces_return = face_classifier.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
    # Faz a varredura na lista de faces detectadas em faces_return
    for (x,y,w,h) in faces_return:

        img = frame.copy()

        # Redimensionando a imagem
        resized_img = imutils.resize(hat, width = w)
        l_image = resized_img.shape[0]
        column_img = w

        dif = 0

        part_alta = l_image // 4

        if y - l_image + part_alta >= 0:
         n_img = img[y - l_image + part_alta: y + part_alta, x: x+w]

        else:
            n_img = img[0: y + part_alta, x: x+w]
            dif = abs(y - l_image + part_alta)

        mask = resized_img[:,:,3]
        mask_not = cv2.bitwise_not(mask)

        mask_and = cv2.bitwise_and(resized_img, resized_img, mask=mask)
        mask_and = mask_and[dif:,:, 0:3]
        mask_frame = cv2.bitwise_and(n_img, n_img, mask=mask_not[dif:,:])

        result = cv2.add(mask_and,mask_frame)

        if y - l_image + part_alta >= 0:
            img[y - l_image + part_alta: y + part_alta, x: x+w] = result
        else: 
            img[0: y + part_alta, x: x+w]
            
        cv2.imshow("result", img)

    # Exibe saida da imagem
    cv2.imshow("normal", frame)
    

    # Atualiza com um novo frame
    rval, frame = video_cap.read()

    # ESC para sair do programa
    key = cv2.waitKey(10)
    if key == 27:
        break

video_cap.release()
cv2.destroyAllWindows()
 