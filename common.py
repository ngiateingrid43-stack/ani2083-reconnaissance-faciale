import cv2
import numpy as np

DET_MODEL = 'models/face_detection_yunet_2023mar.onnx'
REC_MODEL = 'models/face_recognition_sface_2021dec.onnx'

def creer_modeles():
    det = cv2.FaceDetectorYN.create(DET_MODEL, '', (320, 320), 0.8, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(REC_MODEL, '')
    return det, rec

def detecter(det, img):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    return [] if faces is None else faces

def empreinte(rec, img, face):
    e = rec.feature(rec.alignCrop(img, face)).flatten()
    return e / np.linalg.norm(e)
