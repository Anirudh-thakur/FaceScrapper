import cv2
import sys

#Input image
path = r'C:\Users\Anirudh\Desktop\Anirudh\DataScience\Git Projects\FaceScrapper\TestImages\test4.jpg'
image = cv2.imread(path)

#convert to greyscale for ease
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Extract faces
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("Number of faces found :"+str(len(faces)))
face_Num = 1;
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = image[y:y + h, x:x + w]
    print("[INFO] Object found. Saving locally.")
    cv2.imwrite(str(face_Num)+'_faces.jpg', roi_color)
    face_Num = face_Num + 1

status = cv2.imwrite('faces_detected.jpg', image)
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
print(face_Num)




