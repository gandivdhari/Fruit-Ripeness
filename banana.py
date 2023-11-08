import cv2

banacascade=cv2.CascadeClassifier('haarbanana.xml')

videcapture=cv2.VideoCapture(0)

while True:
    ret,frame=videcapture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    banana=banacascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in banana:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        cv2.putText(frame,'Kesel',(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow('detect banana',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


videcapture.release()
cv2.destroyAllWindows()






