from robot_fsm import RobotFSM
from foo_engine import FooEngine

def smoke() -> None:
    fsm = RobotFSM()
    assert fsm.get_state_name() == "Ожидание"
    fsm.on_command("go")
    assert fsm.get_state_name() == "Прокладка пути"
    print("FSM OK")

    engine = FooEngine(broker_host="test.mosquitto.org")
    engine.forward(100); engine.left(100); engine.right(100); engine.stop()
    print("Engine OK (GPIO заглушены, если не на Raspberry Pi)")

if __name__ == "__main__":
    smoke()

