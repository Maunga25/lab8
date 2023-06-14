import time

import cv2
import numpy as np

# Question 1
def image_processing():
    img = cv2.imread('variant-6.png')
    # percentage of stretching
    stretch_percent = 200
    # calculating the new height and width
    nw = int(img.shape[1] * stretch_percent / 100)
    nh = int(img.shape[0] * stretch_percent / 100)
    # resizing the image
    output = cv2.resize(img, (nw, nh))

    cv2.imshow('image', output)


# Question 2
cap = cv2.VideoCapture("sample.mp4")

def object_detector():
    down_points = (640, 550)
    right_hits,i = 0 , 0
    left_hits, i = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        ret, thresh = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            frame=cv2.putText(frame, f'Left hits: {left_hits}' , (30,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA, False )
            frame=cv2.putText(frame, f'Right hits: {right_hits}' , (down_points[0] - 160,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA, False )

            if i % 100 == 0:
                if x + (w // 2) > down_points[0] // 2:
                    right_hits += 1
                elif x + (w // 2) < down_points[0] // 2:
                    left_hits += 1

        cv2.imshow("frame", frame)

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    cap.release()
#доп задание
def additional_task():
    cap = cv2.VideoCapture("sample.mp4")
    down_points = (640, 550)
    right_hits,i = 0 , 0
    left_hits, i = 0, 0
    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame =cv2.resize(frame, down_points, interpolation = cv2.INTER_LINEAR)

        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        ret, thresh = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            cnt = max (contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            img_fly = cv2.resize(fly, (w if x + w < down_points[0] else down_points[0] - x,
                                           h if y + h < down_points[1] else down_points[1] - y))
            alpha_channel, fly_colors = img_fly[:, :, 3] / 255, img_fly[:, :, :3]
            alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
            h_fly, w_fly = img_fly.shape[:2]
            background_subsection = frame[y:(y + h_fly), x:(x + w_fly)]
            composite = background_subsection * (1 - alpha_mask) + fly_colors * alpha_mask
            frame[y:(y + h_fly), x:(x + w_fly)] = composite

            frame=cv2.putText(frame, f'Left hits: {left_hits}' , (30,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA, False )
            frame=cv2.putText(frame, f'Right hits: {right_hits}' , (down_points[0] - 160,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA, False )

            if i % 100 == 0:
                if x + (w // 2) > down_points[0] // 2:
                    right_hits += 1
                elif x + (w // 2) < down_points[0] // 2:
                    left_hits += 1

        cv2.imshow("frame", frame)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    image_processing()
    object_detector()
additional_task()
cv2.waitKey(0)
cv2.destroyAllWindows()


