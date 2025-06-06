# Robot project

# Robot 🚗⚙️

| Компонент | Назначение |
|-----------|------------|
| **FooEngine**           | Принимает MQTT-команды (`/motor/command`), управляет двигателями через GPIO (реальные пины на Raspberry Pi или эмуляция на ПК). |
| **RobotFSM**            | Конечный автомат состояний робота (Idle → Route → Turn → DriveForward → Arrived). |
| **ControlSystem / FooCmdReceiver** | Интерфейс для ручного ввода команд из CLI. |
| **video_player.py**     | Захват видео (камера / файл), поиск цветовых маркеров (🔴 🔵 🟢) и расчёт угла между ними. |
| **Docker Compose**      | Быстрый брокер MQTT (Eclipse Mosquitto) в контейнере: `docker compose up -d`. |
| **test_run.py**         | Smoke-тест FSM + движка на публичном брокере. |

### Быстрый старт

```bash
# зависимости
pip install paho-mqtt gpiozero opencv-python numpy

# поднять брокер
docker compose up -d

# запустить движок
python main.py                # подключится к localhost:1883

# отправить команду
python send_command_example.py forward 1500
