"""
Отображает видеоролик или камеру и вычисляет угол между
линией красный-синий и вектором к зелёному маркеру.
Параметр источника можно передать первым аргументом:
    python video_player.py 0              # камера
    python video_player.py myvideo.mp4    # файл
Требуемые библиотеки: opencv-python, numpy.
"""

import sys
import math
import cv2
import numpy as np
from collections import deque


class ColorDetector:
    """Поиск красного, синего и зелёного пятна, расчёт угла."""
    def __init__(self, history: int = 5) -> None:
        self.last_green_center: tuple[int, int] | None = None
        self.angle_history: deque[float] = deque(maxlen=history)

        # HSV-диапазоны (можно подстроить)
        self.bounds = {
            "red":   (np.array([150, 100, 100]), np.array([165, 255, 255])),
            "blue":  (np.array([90, 100, 100]),  np.array([130, 255, 255])),
            "green": (np.array([40, 100, 100]),  np.array([85, 255, 255])),
        }

    # ---------- внутренние сервисы ----------
    @staticmethod
    def _mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.inRange(hsv, lower, upper)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _center(contour: np.ndarray) -> tuple[int, int]:
        M = cv2.moments(contour)
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    @staticmethod
    def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.degrees(math.acos(cos_theta))

    # ---------- основной метод ----------
    def process(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = {name: self._mask(hsv, *b) for name, b in self.bounds.items()}
        contours = {k: self._largest_contour(m) for k, m in masks.items()}

        # фильтрация шума
        min_area = 500
        for k, c in list(contours.items()):
            if c is not None and cv2.contourArea(c) < min_area:
                contours[k] = None

        # запоминаем зелёный
        if contours["green"] is not None:
            self.last_green_center = self._center(contours["green"])

        # ---------- геометрия ----------
        if contours["red"] is not None and contours["blue"] is not None:
            red_c  = self._center(contours["red"])
            blue_c = self._center(contours["blue"])
            cv2.line(frame, red_c, blue_c, (255, 255, 255), 2)

            green_c = self.last_green_center
            if green_c:
                v_line  = np.array([blue_c[0] - red_c[0],  blue_c[1] - red_c[1]])
                v_green = np.array([green_c[0] - red_c[0], green_c[1] - red_c[1]])
                angle   = self._angle(v_line, v_green)

                self.angle_history.append(angle)
                smooth = float(np.median(self.angle_history))
                cv2.putText(frame, f"{smooth:.1f}°", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # ---------- визуализация ----------
        colors = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}
        for name, contour in contours.items():
            if contour is None:
                continue
            if name == "green":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[name], 2)
            else:
                cv2.drawContours(frame, [contour], -1, colors[name], 2)
        return frame


def main() -> None:
    source = 0 if len(sys.argv) == 1 else sys.argv[1]
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник «{source}»")

    detector = ColorDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Color Detection", detector.process(frame))
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):      # Esc | q
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

