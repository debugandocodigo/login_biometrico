import cv2
import numpy as np
import os
import urllib.request
import ssl

# Desabilita checagem estrita de SSL para evitar falhas de download comuns no pyenv do Mac
ssl._create_default_https_context = ssl._create_unverified_context

# Links oficiais dos modelos de Deep Learning (OpenCV Zoo)
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
ARQUIVO_BIOMETRIA = "biometria_usuario.npy"

def garantir_modelos():
    """Garante que os modelos ONNX de inteligência artificial estejam locais"""
    for url, destino in [(YUNET_URL, YUNET_MODEL), (SFACE_URL, SFACE_MODEL)]:
        if not os.path.exists(destino):
            print(f"📥 Baixando modelo neural essencial ({destino})... Por favor, aguarde.")
            urllib.request.urlretrieve(url, destino)
            print(f"✅ {destino} pronto.")

def inicializar_redes():
    """Instancia os motores de detecção e reconhecimento facial do OpenCV"""
    garantir_modelos()
    # O tamanho inicial (320x320) é atualizado dinamicamente quando a câmera abre
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320))
    reconhecedor = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    return detector, reconhecedor

def iniciar_webcam():
    webcam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not webcam.isOpened():
        webcam = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    return webcam

def cadastrar_rosto():
    detector, reconhecedor = inicializar_redes()
    webcam = iniciar_webcam()
    if not webcam.isOpened():
        print("❌ Erro: Não foi possível acessar a câmera.")
        return

    print("\n📸 [CADASTRO] ATENÇÃO: Clique na janela da câmera com o mouse antes de apertar 'c'!")
    
    while True:
        sucesso, frame = webcam.read()
        if not sucesso:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        detector.setInputSize((w, h))
        _, rostos = detector.detect(frame)
        
        if rostos is not None:
            for rosto in rostos:
                box = rosto[0:4].astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 100, 0), 2)
                cv2.putText(frame, "CLIQUE AQUI e aperte 'c' para salvar", (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        cv2.imshow("Cadastro Biometrico - Deep Learning Nativo", frame)
        
        # Captura o clique do teclado
        tecla = cv2.waitKey(1) & 0xFF
        
        # Aceita tanto 'c' minúsculo quanto 'C' maiúsculo
        if tecla == ord('c') or tecla == ord('C'):
            if rostos is None or len(rostos) == 0:
                print("⚠️ Nenhum rosto detectado! Centralize seu rosto na tela e tente novamente.")
                continue
                
            rosto_alinhado = reconhecedor.alignCrop(frame, rostos[0])
            vetor_facial = reconhecedor.feature(rosto_alinhado)
            
            np.save(ARQUIVO_BIOMETRIA, vetor_facial)
            print(f"✅ Perfeito! Biometria salva com sucesso em '{ARQUIVO_BIOMETRIA}'!")
            break
            
        elif tecla == ord('q') or tecla == ord('Q'):
            print("🔒 Cadastro abortado.")
            break

    webcam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def fazer_login():
    if not os.path.exists(ARQUIVO_BIOMETRIA):
        print("❌ Erro: Nenhuma biometria localizada. Cadastre seu rosto primeiro [Opção 1].")
        return

    # Carrega a biometria cadastrada
    vetor_cadastrado = np.load(ARQUIVO_BIOMETRIA)
    detector, reconhecedor = inicializar_redes()
    
    webcam = iniciar_webcam()
    if not webcam.isOpened():
        return

    # Limiares de aceitação oficiais do modelo SFace:
    # Similaridade de Cosseno >= 0.363 significa que é a mesma pessoa.
    LIMIAR_COSSENO = 0.363
    autenticado = False

    print("\n📸 [LOGIN] Olhe para a câmera para validar seu acesso... (Pressione 'q' para sair)")

    while True:
        sucesso, frame = webcam.read()
        if not sucesso:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        detector.setInputSize((w, h))
        _, rostos = detector.detect(frame)
        
        if rostos is not None:
            for rosto in rostos:
                box = rosto[0:4].astype(int)
                
                # Processa o rosto atual da webcam
                rosto_alinhado = reconhecedor.alignCrop(frame, rosto)
                vetor_atual = reconhecedor.feature(rosto_alinhado)
                
                # Compara os dois vetores usando a métrica de Cosseno
                score_cosseno = reconhecedor.match(vetor_atual, vetor_cadastrado, cv2.FaceRecognizerSF_FR_COSINE)
                
                if score_cosseno >= LIMIAR_COSSENO:
                    autenticado = True
                    cor = (0, 255, 0)
                    txt = f"Acesso Permitido ({round(score_cosseno, 3)})"
                else:
                    cor = (0, 0, 255)
                    txt = f"Desconhecido ({round(score_cosseno, 3)})"
                
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), cor, 2)
                cv2.putText(frame, txt, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        cv2.imshow("Scanner de Login Biometrico", frame)
        
        if autenticado:
            print(f"\n🔓 [LOGIN LIBERADO] Confirmação biométrica positiva! Score: {score_cosseno}")
            cv2.waitKey(2000)
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🔒 Operação cancelada.")
            break

    webcam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    print("="*45)
    print("   SISTEMA DE LOGIN BIOMÉTRICO ATUALIZADO   ")
    print("="*45)
    print("[1] Cadastrar Biometria Facial")
    print("[2] Autenticar / Fazer Login")
    
    opcao = input("Selecione uma opção (1-2): ").strip()
    if opcao == "1":
        cadastrar_rosto()
    elif opcao == "2":
        fazer_login()
    else:
        print("Opção inválida.")