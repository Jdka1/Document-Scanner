import cv2
import numpy as np
from cvzone.ColorModule import ColorFinder



class Camera:
    def __init__(self, camera_port=0):
        self.camera_port = camera_port
        self.capture = cv2.VideoCapture(self.camera_port)
        self.color_finder = ColorFinder(trackBar=False)
        self.HSV_mask_vals = {'hmin': 0, 'smin': 0, 'vmin': 80, 'hmax': 179, 'smax': 100, 'vmax': 255}
        
    def mask_frame(self, img):
        hsv_masked, mask = self.color_finder.update(img, self.HSV_mask_vals)
        return hsv_masked
    
    def get_edges(self, img):
        edges = cv2.Canny(img,220,255)
        return edges
    
    def remove_noise(self, img):
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(img, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        return dilated
        
    def threshold_img(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        return thresh
    
    def get_contours(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours
    
    def get_biggest_rect_contour(self, contours):
        biggest = np.array([])
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 4000:
                perimeter = cv2.arcLength(contour, closed=True)
                approx = cv2. approxPolyDP(contour, 0.02 * perimeter, closed=True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest
        
    def reorder(self, pts):
        pts = pts.reshape((4,2))
        pts_new = np.zeros((4,1,2), dtype=np.float32)
        add = pts.sum(1)
        
        pts_new[1] = pts[np.argmin(add)]
        pts_new[3] = pts[np.argmax(add)]
        diff = np.diff(pts, axis=1)
        pts_new[0] = pts[np.argmin(diff)]
        pts_new[2] = pts[np.argmax(diff)]
        return pts_new
        
    def warp_img(self, img, corners, width, height):
        pts1 = np.float32([
            corners[0][0],
            corners[1][0],
            corners[2][0],
            corners[3][0]
        ])
        
        pts1 = self.reorder(pts1)
        
        
        pts2 = np.float32([
            [width,0],
            [0,0],
            [0,height],
            [width,height]
        ])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        output = cv2.warpPerspective(img, matrix, (width, height))
        return output
    
    def draw_corners(self, img, contour):
        pts1 = np.array([i[0] for i in contour])
        output = cv2.circle(img, (pts1[0][0], pts1[0][1]), 8, (0,255,0), -1)
        for i in pts1[1:]:
            output = cv2.circle(output, (i[0], i[1]), 8, (0,255,0), -1)
        return output
    
    def contrast(self, img):
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img
        
    def gen_video(self):
        while True:
            success, frame = self.capture.read()
            
            try:
                masked = self.mask_frame(frame)
                thresh = self.threshold_img(masked)
                contours = self.get_contours(thresh)
                
                if contours:
                    biggest = self.get_biggest_rect_contour(contours)
                    try:
                        warped = self.warp_img(frame, biggest, width=2550, height=3300)
                        contrast = self.contrast(warped)
                        cv2.imshow('warped', contrast)
                    except:
                        print('No document found.')
            except:
                try:
                    cv2.imshow('frame', frame)
                except:
                    pass
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                try:
                    cv2.imwrite('scan.png', contrast)
                except Exception as e:
                    print(e)
