import cv2, os, numpy as np
from common import *

det, rec = creer_modeles()
noms, embs = [], []
racine = os.path.join('data', 'people')

for nom in sorted(os.listdir(racine)):
    for f in os.listdir(os.path.join(racine, nom)):
        img = cv2.imread(os.path.join(racine, nom, f))
        if img is None:
            continue
        faces = detecter(det, img)
        if len(faces) == 0:
            print('Aucun visage :', nom, f)
            continue
        face = max(faces, key=lambda x: x[2] * x[3])
        embs.append(empreinte(rec, img, face))
        noms.append(nom)

np.savez('embeddings.npz', noms=np.array(noms), embs=np.array(embs))
print(len(noms), 'visages enregistrés')
