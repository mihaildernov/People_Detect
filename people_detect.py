import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # перевод кадров в черно-белую градацию

    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # фильтрация лишних контуров

    rects, weights = hog.detectMultiScale(blur)

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Human Detection', frame)

    print('Людей обнаружено: ', len(rects))

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
