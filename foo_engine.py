import os
import threading
import time
from typing import Optional

from robot_fsm import RobotFSM
from engine_base import AEngine


USE_GPIO = False
try:
    # До импорта gpiozero указываем, что нужен его mock
    os.environ.setdefault("GPIOZERO_PIN_FACTORY", "mock")

    from gpiozero import OutputDevice
    from gpiozero.devices import Device
    from gpiozero.pins.mock import MockFactory
    from gpiozero.exc import BadPinFactory

    # Если всё-таки загрузилась «неправильная» фабрика, принудительно ставим MockFactory
    if not isinstance(Device.pin_factory, MockFactory):
        Device.pin_factory = MockFactory()

    # Пробуем создать тестовый пин — если успех, значит gpiozero доступна
    _test = OutputDevice(4)
    _test.close()
    USE_GPIO = True
    print("GPIOZero работает в mock-режиме.")
except Exception as e:
    # Полная заглушка, если gpiozero вообще недоступна
    class OutputDevice:
        def __init__(self, pin: int) -> None:
            self.pin = pin

        def on(self) -> None: print(f"[GPIO {self.pin}] ON")

        def off(self) -> None: print(f"[GPIO {self.pin}] OFF")


    print(f"gpiozero недоступна ({e!s}); использую stub-пины.")

import paho.mqtt.client as mqtt

class FooEngine(AEngine):
    TOPIC = "/motor/command"

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883) -> None:
        self.fsm   = RobotFSM()
        self.mutex = threading.Lock()

        # GPIO-пины двигателя
        self.pin_a1 = OutputDevice(12)
        self.pin_a2 = OutputDevice(13)
        self.pin_b1 = OutputDevice(20)
        self.pin_b2 = OutputDevice(21)

        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker_host, broker_port, keepalive=60)

        threading.Thread(target=self.client.loop_forever,
                         daemon=True).start()

    # ---------- MQTT ----------
    def _on_connect(self, client, userdata, flags, rc) -> None:
        print(f"MQTT connected rc={rc}")
        client.subscribe(self.TOPIC)

    def _on_message(self, client, userdata, msg) -> None:
        payload = msg.payload.decode()
        print(f"MQTT << {payload}")
        self._dispatch(payload)

    # ---------- AEngine ----------
    def forward(self, ms: int = 0) -> None:
        with self.mutex:
            print("Engine: forward")
            self._set_pins(True, True, False, False)
            self.fsm.on_command("forward")
        self._sleep(ms)

    def left(self, ms: int = 0) -> None:
        with self.mutex:
            print("Engine: left")
            self._set_pins(False, True, True, False)
            self.fsm.on_command("turn")
        self._sleep(ms)

    def right(self, ms: int = 0) -> None:
        with self.mutex:
            print("Engine: right")
            self._set_pins(True, False, False, True)
            self.fsm.on_command("turn")
        self._sleep(ms)

    def stop(self) -> None:
        """Останавливает моторы и закрывает MQTT-соединение."""
        with self.mutex:
            print("Engine: stop")
            self._set_pins(False, False, False, False)
            self.fsm.on_command("idle")
        # корректное завершение MQTT
        try:
            self.client.disconnect()
        except Exception:
            pass

    # ---------- helpers ----------
    def _dispatch(self, cmd: str) -> None:
        cmd = cmd.strip().lower()
        if cmd.startswith("forward"):
            parts = cmd.split()
            self.forward(int(parts[1]) if len(parts) > 1 else 0)
        elif cmd in ("left", "right", "stop", "idle"):
            getattr(self, cmd)()
        else:
            print(f"[FooEngine] Unknown command: {cmd}")

    def _set_pins(self, a1, a2, b1, b2) -> None:
        (self.pin_a1.on() if a1 else self.pin_a1.off())
        (self.pin_a2.on() if a2 else self.pin_a2.off())
        (self.pin_b1.on() if b1 else self.pin_b1.off())
        (self.pin_b2.on() if b2 else self.pin_b2.off())

    @staticmethod
    def _sleep(ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000)