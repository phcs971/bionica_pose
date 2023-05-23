import cv2
import numpy as np
import os

OUTPUT_PATH = 'assets/output_video.mp4'
INPUT_PATH = 'assets/video.mp4'

GRAY_THRESHOLD = 210

GROUP_DISTANCE_THRESHOLD = 45

class CornerGroup:
    def __init__(self, corner):
        self.corners = [corner]

    def addPoint(self, corner):
        self.corners.append(corner)

    def center(self):
        return np.mean(self.corners, axis=0).astype(int)

    def distance(self, corner):
        return np.linalg.norm(corner - self.center())


def v1(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[0:510, 0:352]
    _,thresh = cv2.threshold(gray,GRAY_THRESHOLD,255, cv2.THRESH_BINARY)

    corners = cv2.goodFeaturesToTrack(thresh, 100, 0.2, 10)
    corners = np.intp(corners)

    grouped_corners = [CornerGroup(corners[0])]

    distance_threshold = GROUP_DISTANCE_THRESHOLD

    for corner in corners[1:]:
        added = False
        for group in grouped_corners:
            if group.distance(corner) < distance_threshold:
                added = True
                group.addPoint(corner)
                break
        if not added:
            grouped_corners.append(CornerGroup(corner))

    grouped_corners = [group for group in grouped_corners if len(group.corners) > 2]

    circles = [group.center() for group in grouped_corners]
    for i in circles:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 3, 255, -1)
        cv2.drawMarker(frame, (x, y), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=3)

    return frame

def main():
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)


    cap = cv2.VideoCapture(INPUT_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = cv2.VideoWriter_fourcc(*'avc1')

    success, frame = cap.read()
    result =  cv2.VideoWriter(OUTPUT_PATH,codec, fps, frameSize)

    while success:
        output = v1(frame)
        result.write(output)

        success, frame = cap.read()
    
    result.release()
    
if __name__ == '__main__':
    main()