python3 -c "import cv2
for idx in [0, 1]:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    print(f'video{idx}: opened={cap.isOpened()}')
    cap.release()"
