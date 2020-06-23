#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os


# In[ ]:


face_class = cv2.CascadeClassifier('C://Users//KIIT//Desktop//mlops//Face Detection using VGG16//Dataset//haarcascade_frontalface_default.xml')


# In[ ]:


def face_extractor (photo):   
    gphoto = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    detected = face_class.detectMultiScale(gphoto)
    
    if detected == ():
        return None 
    else :
        ( x , y , w , h ) = detected[0]
        cphoto = photo [ y:y+h , x:x+w ]
        return cphoto


# In[ ]:


name = input("Enter your name here : ")


# In[ ]:


directory = name
parent_dir = "C://Users//KIIT//Desktop//mlops//Face Detection using VGG16//Dataset//train//"
path = os.path.join(parent_dir, directory) 
os.mkdir(path) 
parent_dir = "C://Users//KIIT//Desktop//mlops//Face Detection using VGG16//Dataset//test//"
path = os.path.join(parent_dir, directory) 
os.mkdir(path)


# In[ ]:


cap = cv2.VideoCapture(0)
i=0
while True :
    
    status, photo = cap.read()
    
    if face_extractor (photo) is not None :
        cphoto = face_extractor ( photo)
        i=int(i)+1
        face = cv2.resize( cphoto , (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite("C://Users//KIIT//Desktop//mlops//Face Detection using VGG16//Dataset//train//"+name+"//"+name+str(i)
                    +".jpg" , face)

        cv2.putText(face , str(i) , (100, 100) ,cv2.FONT_HERSHEY_SIMPLEX, 2 , [255,0,0],3)
        cv2.putText(face , "train set" , (0, 30) ,cv2.FONT_HERSHEY_SIMPLEX, 0.7 , [255,0,0],3)
        cv2.imshow("Face cropper",face)
        
    else:
        print("Face not found")
        pass
    
    if cv2.waitKey(1) == 13 or i == 70: #13 is the Enter Key
        break
        
cap.release()       
cv2.destroyAllWindows()


# In[ ]:


cap = cv2.VideoCapture(0)
i=0
while True :
    
    status, photo = cap.read()
    
    if face_extractor (photo) is not None :
        cphoto = face_extractor (photo)
        i=int(i)+1
        face = cv2.resize( cphoto , (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite("C://Users//KIIT//Desktop//mlops//Face Detection using VGG16//Dataset//test//"+name+"//"+name+str(i)+".jpg" , face)

        cv2.putText(face , str(i) , (100, 100) ,cv2.FONT_HERSHEY_SIMPLEX, 2 , [255,0,0],3)
        cv2.putText(face , "test set" , (0, 30) ,cv2.FONT_HERSHEY_SIMPLEX, 0.7 , [255,0,0],3)
        cv2.imshow("Face cropper",face)
        
    else:
        print("Face not found")
        pass
    
    if cv2.waitKey(1) == 13 or i == 30: #13 is the Enter Key
        break
        
cap.release()       
cv2.destroyAllWindows()

print("Sample Collection Completed")


# In[ ]:
