from keras.models import load_model
import cv2
import numpy as np

model         = load_model('models/mnistCNN_num.model')
digits        = ['0', '1', '2', '3', '4', '5', '6', '7','8','9']
image         = cv2.imread('./images/0013.jpg');#cv2.imshow('image',image);cv2.waitKey(0)
gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#;cv2.imshow('gray',gray);cv2.waitKey(0)
blur          = cv2.GaussianBlur(gray, (5, 5), 0)#;cv2.imshow('blur',blur);cv2.waitKey(0)
thresh        = cv2.adaptiveThreshold(blur ,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,39,13)#;cv2.imshow('thresh',thresh);cv2.waitKey(0)
thresh_inv    = 255 - thresh #;cv2.imshow('thresh_inv',thresh_inv);cv2.waitKey(0)
cnts          = cv2.findContours(thresh_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts          = sorted(cnts, key=cv2.contourArea, reverse=True)

for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    ar        = w / float(h)
    crWidth   = w / float(image.shape[1])
    crHeight  = h / float(image.shape[0])
    #if ar > 0.1 and ar < 5 and crWidth > 0.01 and crWidth < 1 and crHeight > 0.05 and crHeight < 1:
    if ar > 0.05 and ar < 10 and crWidth > 0.05 and crWidth < 1 and crHeight > 0.2 and crHeight < 1:
    	cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)
    	roi       = thresh_inv[y:y + h, x:x + w]#;cv2.imshow('roi',roi);cv2.waitKey(0)
    	roi       = cv2.resize(roi, (28, 28))#;cv2.imshow('roi',roi);cv2.waitKey(0)
    	predicted = np.argmax(model.predict(roi.reshape(1, 28, 28, 1)))
    	cv2.putText(image, digits[predicted], (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

cv2.imshow('image',image)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)

