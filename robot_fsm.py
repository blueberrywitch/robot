from enum import Enum, auto

class State(Enum):
    IDLE = auto()
    ROUTE = auto()
    TURN = auto()
    DRIVE_FORWARD = auto()
    ARRIVED = auto()

class RobotFSM:
    def __init__(self) -> None:
        self.state = State.IDLE

    def get_state_name(self) -> str:
        names = {
            State.IDLE:          "Ожидание",
            State.ROUTE:         "Прокладка пути",
            State.TURN:          "Поворот",
            State.DRIVE_FORWARD: "Движение вперёд",
            State.ARRIVED:       "Прибытие"
        }
        return names[self.state]

    def on_command(self, cmd: str) -> None:
        cmd = cmd.strip().lower()
        if cmd == "go":
            self.state = State.ROUTE
        elif cmd.startswith("turn"):
            self.state = State.TURN
        elif cmd.startswith("forward"):
            self.state = State.DRIVE_FORWARD
        elif cmd == "arrived":
            self.state = State.ARRIVED
        elif cmd == "idle":
            self.state = State.IDLE

