import cv2, os, sys

nom = sys.argv[1]
dossier = os.path.join('data', 'people', nom)
os.makedirs(dossier, exist_ok=True)
n = len(os.listdir(dossier))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    vue = frame.copy()
    cv2.putText(vue, f'{nom}: {n} images | ESPACE=photo, Q=quitter', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Collecte', vue)
    k = cv2.waitKey(1) & 0xFF
    if k == 32:
        cv2.imwrite(os.path.join(dossier, f'{nom}_{n:03d}.jpg'), frame)
        n += 1
    elif k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
Partagez le dossier data/ par Google Drive privé : jamais sur GitHub.
7.3 Interface PyQt5 du Projet 1
Fonctions : Ouvrir la webcam, Arrêter la webcam, Charger une image, Enregistrer le résultat, curseur du seuil de confiance, affichage des noms et des scores. Elle utilise common.py et embeddings.npz : lancez d’abord python build_db.py.
app_visage.py
import sys, os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from common import creer_modeles, detecter, empreinte

if not os.path.exists('embeddings.npz'):
    sys.exit('embeddings.npz introuvable : lancez d abord python build_db.py')

data = np.load('embeddings.npz')
noms, embs = data['noms'], data['embs']
centres = {}
for p in sorted(set(noms)):
    c = embs[noms == p].mean(axis=0)
    centres[p] = c / np.linalg.norm(c)
det, rec = creer_modeles()

def identifier(img, seuil):
    resultats = []
    for face in detecter(det, img):
        e = empreinte(rec, img, face)
        scores = {p: float(e @ c) for p, c in centres.items()}
        nom, s = max(scores.items(), key=lambda x: x[1])
        if s < seuil:
            nom = 'Inconnu'
        x, y, w, h = face[:4].astype(int)
        couleur = (0, 200, 0) if nom != 'Inconnu' else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), couleur, 2)
        cv2.putText(img, f'{nom} {s:.2f}', (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
        resultats.append((nom, s))
    return img, resultats

class Fenetre(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Reconnaissance faciale - ANI-2083')
        self.cap = None
        self.dernier = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.lire_image)

        self.ecran = QLabel('Ouvrez la webcam ou chargez une image')
        self.ecran.setFixedSize(640, 480)
        self.ecran.setAlignment(Qt.AlignCenter)
        self.ecran.setStyleSheet('background: #222; color: white;')
        self.statut = QLabel('Prêt.')
        self.statut.setStyleSheet('font-size: 15px;')
        self.etiq = QLabel('Seuil de confiance : 0.36')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(20, 70)
        self.slider.setValue(36)
        self.slider.valueChanged.connect(
            lambda v: self.etiq.setText(f'Seuil de confiance : {v / 100:.2f}'))

        b_web = QPushButton('Ouvrir la webcam')
        b_stop = QPushButton('Arrêter la webcam')
        b_img = QPushButton('Charger une image')
        b_enr = QPushButton('Enregistrer le résultat')
        b_web.clicked.connect(self.ouvrir_webcam)
        b_stop.clicked.connect(self.arreter_webcam)
        b_img.clicked.connect(self.charger_image)
        b_enr.clicked.connect(self.enregistrer)

        boutons = QHBoxLayout()
        for b in (b_web, b_stop, b_img, b_enr):
            boutons.addWidget(b)
        seuil = QHBoxLayout()
        seuil.addWidget(self.etiq)
        seuil.addWidget(self.slider)
        mise = QVBoxLayout(self)
        mise.addWidget(self.ecran)
        mise.addLayout(boutons)
        mise.addLayout(seuil)
        mise.addWidget(self.statut)

    def afficher(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(self.ecran.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation)
        self.ecran.setPixmap(pix)

    def traiter(self, img):
        img, res = identifier(img, self.slider.value() / 100)
        self.dernier = img.copy()
        self.afficher(img)
        if res:
            self.statut.setText(' | '.join(f'{n} (confiance {s:.2f})' for n, s in res))
        else:
            self.statut.setText('Aucun visage détecté.')

    def ouvrir_webcam(self):
        if self.cap is not None:
            return
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            QMessageBox.warning(self, 'Erreur', 'Impossible d ouvrir la webcam.')
            return
        self.cap = cap
        self.timer.start(30)

    def arreter_webcam(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def lire_image(self):
        ok, frame = self.cap.read()
        if not ok:
            self.arreter_webcam()
            self.statut.setText('Lecture de la webcam impossible.')
            return
        self.traiter(frame)

    def charger_image(self):
        chemin, _ = QFileDialog.getOpenFileName(self, 'Choisir une image', '',
                                                'Images (*.jpg *.jpeg *.png *.bmp)')
        if not chemin:
            return
        self.arreter_webcam()
        img = cv2.imdecode(np.fromfile(chemin, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, 'Erreur', 'Image illisible.')
            return
        self.traiter(img)

    def enregistrer(self):
        if self.dernier is None:
            QMessageBox.information(self, 'Info', 'Rien à enregistrer pour le moment.')
            return
        chemin, _ = QFileDialog.getSaveFileName(self, 'Enregistrer le résultat',
                                                'resultat.png', 'PNG (*.png)')
        if chemin:
            ok, buf = cv2.imencode('.png', self.dernier)
            if ok:
                buf.tofile(chemin)

    def closeEvent(self, event):
        self.arreter_webcam()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    f = Fenetre()
    f.show()
    sys.exit(app.exec_())
