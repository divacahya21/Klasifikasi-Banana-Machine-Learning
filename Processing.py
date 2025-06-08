import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# Load YOLOv8 model (pastikan yolov8s.pt ada di root)
yolo_model = YOLO("yolov8s.pt")

def crop_banana_yolo(gambar_pil, conf=0.4):
    """
    Deteksi dan crop semua buah pisang dari gambar menggunakan YOLO.
    """
    hasil_crop = []
    hasil_deteksi = yolo_model.predict(source=gambar_pil, conf=conf, verbose=False)
    for hasil in hasil_deteksi:
        for box in hasil.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = gambar_pil.crop((x1, y1, x2, y2))
            hasil_crop.append(crop)
    return hasil_crop

def fitur_histogram_hue(img_pil, bins=64):
    """
    Ekstraksi histogram channel Hue dari HSV.
    """
    img_hsv = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2HSV)
    hue_channel = img_hsv[:, :, 0]
    hist = cv2.calcHist([hue_channel], [0], None, [bins], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def fitur_glcm(img_gray):
    """
    Ekstraksi fitur tekstur dari citra grayscale menggunakan GLCM.
    """
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    fitur = [graycoprops(glcm, prop)[0, 0] for prop in props]
    return fitur

def fitur_statistik_hue(img_pil):
    """
    Hitung mean, std, skewness, dan kurtosis dari channel Hue.
    """
    img_hsv = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2HSV)
    hue_channel = img_hsv[:, :, 0].flatten()
    mean_hue = np.mean(hue_channel)
    std_hue = np.std(hue_channel)
    skewness = skew(hue_channel)
    kurt = kurtosis(hue_channel)
    return [mean_hue, std_hue, skewness, kurt]

def ekstrak_fitur_lengkap(img_pil):
    """
    Gabungkan semua fitur: histogram hue, GLCM, dan statistik hue.
    """
    hist = fitur_histogram_hue(img_pil)
    gray = np.array(img_pil.convert("L"))
    glcm = fitur_glcm(gray)
    stat = fitur_statistik_hue(img_pil)
    return np.concatenate([hist, glcm, stat])
