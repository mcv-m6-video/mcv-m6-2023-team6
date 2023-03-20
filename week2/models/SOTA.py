import numpy as np
import cv2

from models.BaseModel import BaseModel


class SOTA (BaseModel):
    def __init__(self, video_path, num_frames, p, checkpoint=None, n_jobs=-1, method='MOG'):
        super().__init__(video_path, num_frames, checkpoint)

        if method == 'MOG':
            self.method = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=2, backgroundRatio=0.7)
        elif method == 'MOG2':
            self.method = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=36, detectShadows=True)
        elif method == 'LSBP':
            self.method = cv2.bgsegm.createBackgroundSubtractorLSBP()
        elif method == 'ViBe': # Extra SOTA --> variant of the ViBe algorithm called the Count-based Neighbors Technique (CNT).
            self.method = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True, isParallel=False)
        else:
            raise Exception('Invalid method')
    
    def compute_next_foreground(self):
        ok, frame = self.cap.read()

        if not ok:
            return None
    
        fgmask = self.method.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        return fgmask, frame

    def model_background(self):
        frame = self.cap.read()
        i = 1
        while frame is not None and i < self.num_frames:

            #fgmask = self.method.apply(frame)
            #cv2.imshow('frame',fgmask)
            #k = cv2.waitKey(30) & 0xff
            #if k == 27:
            #    break
        
            frame = self.cap.read()
            i += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        return i
    

