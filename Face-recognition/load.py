import cv2
import face_recognition

# load the image and the test image
imgSanjar = face_recognition.load_image_file('Images/Sanjar.jpg')
imgSanjar = cv2.cvtColor(imgSanjar, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/Diyar.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# find the face locations
faceLoc = face_recognition.face_locations(imgSanjar)[0]
encodeElon = face_recognition.face_encodings(imgSanjar)[0]
cv2.rectangle(imgSanjar, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# test image face locations
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# compare the images
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# show the detected face recognitions
cv2.imshow('Sanjar', imgSanjar)
cv2.imshow('Sanjar Test', imgTest)
cv2.waitKey(0)