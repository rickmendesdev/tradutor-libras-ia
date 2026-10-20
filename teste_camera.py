import cv2

from landmarks import Detector, desenhar_maos, desenhar_corpo

# Prática:

# Crie um ambiente virtual e rode pip install -r requirements.txt.
# Rode python teste_camera.py. Você deve ver o esqueleto verde nas mãos, os pontos azuis no corpo e o texto “Maos: X | Corpo: OK”. Fique com o tronco e os ombros inteiros no quadro. Se aparecer “NAO DETECTADO”, afaste a câmera.
# Abra o coletar_dados.py e ajuste PESSOA e a lista ACOES. A lista que deixei (feliz, triste, raiva, medo, amor, parado) é só um exemplo. Troquem pelos sinais que vocês validaram com o dicionário do INES ou com alguém que sabe Libras.
# Rode uma coleta de teste com poucas repetições e olhe um arquivo:

detector = Detector()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        print("Camera ignorada ou nao encontrada. Verifique se o Iriun mostra imagem.")
        break

    res_maos, res_pose = detector.processar(frame)
    desenhar_maos(frame, res_maos)
    desenhar_corpo(frame, res_pose)

    n_maos = len(res_maos.hand_landmarks) if res_maos.hand_landmarks else 0
    corpo_ok = bool(res_pose.pose_landmarks)
    cv2.putText(frame, f"Maos: {n_maos} | Corpo: {'OK' if corpo_ok else 'NAO DETECTADO'}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Teste MediaPipe - Aperte Q para sair', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
