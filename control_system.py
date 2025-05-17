class ControlSystem:
    """Читает команды из ACmdReceiver и управляет AEngine."""
    def __init__(self, engine: 'AEngine', receiver: 'ACmdReceiver') -> None:
        self.engine   = engine
        self.receiver = receiver

    def run(self) -> None:
        while True:
            cmd = self.receiver.receive()
            if not cmd:
                continue

            cmd_low = cmd.strip().lower()
            if cmd_low == "exit":
                print("Exit command received. Stopping control system.")
                self.engine.stop()
                break
            elif cmd_low.startswith("forward"):
                parts = cmd_low.split()
                ms = int(parts[1]) if len(parts) > 1 else 0
                self.engine.forward(ms)
            elif cmd_low == "left":
                self.engine.left()
            elif cmd_low == "right":
                self.engine.right()
            elif cmd_low == "stop":
                self.engine.stop()
            else:
                print(f"Неизвестная команда: {cmd}")

