import cv2
import time
import threading
import config

class Camera:
    def __init__(self):
        self.index    = config.CAMERA_INDEX
        self.width    = config.CAMERA_WIDTH
        self.height   = config.CAMERA_HEIGHT
        self.interval = config.CAPTURE_INTERVAL

        self._frame     = None
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None
        self._last_capture = 0

    def start(self):
        """Inicia a thread de captura em background."""
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print("[Camera] Erro: não foi possível abrir a câmera.")
            return False

        self._running = True
        self._thread  = threading.Thread(target=self._loop, args=(cap,), daemon=True)
        self._thread.start()
        print("[Camera] Thread de captura iniciada.")
        return True

    def _loop(self, cap):
        """Loop interno que mantém o frame mais recente sempre disponível."""
        while self._running:
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = rgb
            time.sleep(0.01)  # ~100fps interno, sem sobrecarregar
        cap.release()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def should_capture(self):
        """Retorna True se já passou o intervalo desde a última captura."""
        return (time.time() - self._last_capture) >= self.interval

    def capture(self):
        """
        Retorna o frame mais recente em RGB, ou None se ainda não disponível.
        Não abre nem fecha a câmera — apenas lê o buffer.
        """
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()

        self._last_capture = time.time()
        return frame