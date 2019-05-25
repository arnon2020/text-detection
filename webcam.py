#print(hex(id(image))) #print memory address
import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import time

def make_digit(roi,size=(28,28),multiply=1.2):
    roi_h, roi_w = roi.shape
    if roi_h > roi_w:
        roi_h = int(roi_h * multiply)
        roi_x = int(roi_h / 2) - int(roi.shape[1] / 2)
        roi_y = int(roi_h / 2) - int(roi.shape[0] / 2)
        zero = np.zeros((roi_h, roi_h))
    else:
        roi_w = int(roi_w * multiply)
        roi_x = int(roi_w / 2) - int(roi.shape[1] / 2)
        roi_y = int(roi_w / 2) - int(roi.shape[0] / 2)
        zero = np.zeros((roi_w, roi_w))
    #plt.imshow(zero);plt.show()
    zero[roi_y:roi_y + roi.shape[0], roi_x:roi_x + roi.shape[1]] = roi #;plt.imshow(zero);plt.show()
    zero = cv2.resize(zero, size , interpolation=cv2.INTER_AREA) #;plt.imshow(zero);plt.show()
    return zero

#size image
width = 640
height = 480

im_W = int(width / 2)  # im_width_center
im_H = int(height / 2)  # im_height_center

#Green Box Size
box_W = 400   # box_width_half
box_H = 400  # box_height_half
box_W = int(box_W / 2)
box_H = int(box_H / 2)

GB_x1 = im_W - box_W
GB_x2 = im_W + box_W
GB_y1 = im_H - box_H
GB_y2 = im_H + box_H
GB_Start = (GB_x1, GB_y1)
GB_Stop = (GB_x2, GB_y2)

#set camera
cam    = 1
cap    = cv2.VideoCapture(cam)

#load model
model  = load_model('models/mnistCNN_num.model')

digits = ['0', '1', '2', '3', '4', '5', '6', '7','8','9']

fps = 0
count = 0
pretime = time.time()

while(True):
	ret, image          = cap.read()
	try:
		#image           = cv2.imread('./images/0013.jpg')
		#image           = cv2.resize(image,(640,480))
		image2          = cv2.rectangle(image.copy(), GB_Start, GB_Stop, (0, 255, 0), 3)  # Green BOX
		im_GB           = image[GB_y1:GB_y2, GB_x1:GB_x2].copy()
		gray_GB         = cv2.cvtColor(im_GB, cv2.COLOR_BGR2GRAY)
		blur_GB         = cv2.medianBlur(gray_GB,1)
		thresh_GB       = cv2.adaptiveThreshold(blur_GB ,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,39,13)#;plt.imshow(thresh_GB);plt.show()
		thresh_GB_inv   = 255 - thresh_GB#;plt.imshow(thresh_GB_inv);plt.show()
		cnts_GB     = cv2.findContours(thresh_GB_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

		for c in cnts_GB:
			(x, y, w, h) = cv2.boundingRect(c)
			ar           = w / float(h)
			crWidth      = w / float(im_GB.shape[1])
			crHeight     = h / float(im_GB.shape[0])
			if ar > 0.1 and ar < 5 and crWidth > 0.01 and crWidth < 1 and crHeight > 0.1 and crHeight < 1:
				roi = thresh_GB_inv[y:y + h, x:x + w]#;plt.imshow(roi);plt.show()
				roi = make_digit(roi)
				output = model.predict(roi.reshape(1, 28, 28, 1))[0];print(output)
				predicted = np.argmax(output)
				percentage = int(output[predicted] * 100)
				if percentage > 90:
					cv2.rectangle(image2, (x + GB_x1, y + GB_y1), (x + GB_x1 + w, y + GB_y1 + h), (0, 0, 255), 2)
					cv2.putText(image2, digits[predicted], (GB_x1+x, GB_y1+y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
		count += 1
		if time.time()-pretime >= 1:
			fps = count
			count = 0
			pretime = time.time()

		cv2.putText(image2, 'fps:{:}'.format(fps), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
		cv2.imshow('image', image2)
		#cv2.waitKey(0)
		#cv2.imshow('im_GB', im_GB)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	except:
		image = cv2.imread('./images/cam_error.jpg')
		cv2.imshow('image', image)
		cv2.waitKey(0)
		cap    = cv2.VideoCapture(cam)

cap.release()
cv2.destroyAllWindows()