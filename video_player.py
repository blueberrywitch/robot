import cv2
import numpy as np
import math
from collections import deque
import sys
from dataclasses import dataclass

@dataclass
class HSVRange:
    lower: np.ndarray
    upper: np.ndarray

class ColorDetector:
    """
    Детектирует красный, синий и зелёный маркеры и
    вычисляет угол между отрезком (красный–синий) и вектором на зелёный.
    """
    def __init__(self,
                 history: int = 10,
                 min_area: int = 400,
                 kernel_size: int = 5) -> None:
        self.last_green_center: tuple[int, int] | None = None
        self.angle_history: deque[float] = deque(maxlen=history)
        self.min_area = min_area
        self.kernel   = np.ones((kernel_size, kernel_size), np.uint8)

        # HSV-диапазоны
        self.bounds: dict[str, list[HSVRange]] = {
            "red": [
                HSVRange(np.array([0,   100, 100]), np.array([10, 255, 255])),
                HSVRange(np.array([170, 100, 100]), np.array([180, 255, 255])),
            ],
            "blue": [
                HSVRange(np.array([90,  100, 100]), np.array([130, 255, 255]))
            ],
            "green": [
                HSVRange(np.array([40,  100, 100]), np.array([85, 255, 255]))
            ],
        }

        self.vis_colors = {"red": (0, 0, 255),
                           "blue": (255, 0, 0),
                           "green": (0, 255, 0)}

    # ---------- утилиты ----------
    def _mask(self, hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            single = cv2.inRange(hsv, r.lower, r.upper)
            mask   = cv2.bitwise_or(mask, single)
        # закрываем дырки, затем убираем шум
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        return mask

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(cnts, key=cv2.contourArea) if cnts else None

    @staticmethod
    def _center(contour: np.ndarray) -> tuple[int, int] | None:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    @staticmethod
    def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return math.degrees(math.acos(cos_theta))

    # ---------- основной метод ----------
    def process(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks     = {k: self._mask(hsv, rngs) for k, rngs in self.bounds.items()}
        contours  = {k: self._largest_contour(m) for k, m in masks.items()}

        # фильтрация по площади
        for k, c in contours.items():
            if c is not None and cv2.contourArea(c) < self.min_area:
                contours[k] = None

        # запоминаем зелёный
        if contours["green"] is not None:
            self.last_green_center = self._center(contours["green"])

        # геометрия
        red_c   = self._center(contours["red"])  if contours["red"]  is not None else None
        blue_c  = self._center(contours["blue"]) if contours["blue"] is not None else None
        green_c = self.last_green_center

        if red_c and blue_c:
            cv2.line(frame, red_c, blue_c, (255, 255, 255), 2)
            if green_c:
                v_line  = np.subtract(blue_c, red_c)
                v_green = np.subtract(green_c, red_c)
                angle   = self._angle(v_line, v_green)

                # EMA-сглаживание
                alpha   = 0.2
                prev    = self.angle_history[-1] if self.angle_history else angle
                smooth  = alpha * angle + (1 - alpha) * prev
                self.angle_history.append(smooth)

                cv2.putText(frame, f"{smooth:.1f}°", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (255, 255, 255), 2, cv2.LINE_AA)

        # визуализация контуров
        for name, contour in contours.items():
            if contour is None:
                continue
            if name == "green":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              self.vis_colors[name], 2)
            else:
                cv2.drawContours(frame, [contour], -1,
                                 self.vis_colors[name], 2)
        return frame

# ---------- точка входа ----------
def main() -> None:
    source = 0 if len(sys.argv) == 1 else sys.argv[1]
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник «{source}»")

    detector = ColorDetector(history=10, min_area=400)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output = detector.process(frame)
        cv2.imshow("Color Detection", output)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
