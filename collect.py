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
