import cv2, sys, numpy as np
from common import *

SEUIL = 0.363   # à ajuster avec l'évaluation

data = np.load('embeddings.npz')
noms, embs = data['noms'], data['embs']
centres = {}
for p in sorted(set(noms)):
    c = embs[noms == p].mean(axis=0)
    centres[p] = c / np.linalg.norm(c)

det, rec = creer_modeles()

def identifier(img):
    for face in detecter(det, img):
        e = empreinte(rec, img, face)
        scores = {p: float(e @ c) for p, c in centres.items()}
        nom, s = max(scores.items(), key=lambda x: x[1])
        if s < SEUIL:
            nom = 'Inconnu'
        x, y, w, h = face[:4].astype(int)
        couleur = (0, 255, 0) if nom != 'Inconnu' else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), couleur, 2)
        cv2.putText(img, f'{nom} {s:.2f}', (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
    return img

if len(sys.argv) > 1:
    img = identifier(cv2.imread(sys.argv[1]))
    cv2.imshow('Resultat', img)
    cv2.waitKey(0)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow('Reconnaissance', identifier(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
cv2.destroyAllWindows()
