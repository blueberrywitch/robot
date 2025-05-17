import time
from foo_engine import FooEngine

def main() -> None:
    engine = FooEngine()            # подписываемся на MQTT
    print("Приложение запущено. Ctrl+C для выхода.")
    try:
        while True:
            time.sleep(0.1)         # экономим CPU
    except KeyboardInterrupt:
        engine.stop()

if __name__ == "__main__":
    main()
