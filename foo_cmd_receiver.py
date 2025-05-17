from cmd_receiver_base import ACmdReceiver

class FooCmdReceiver(ACmdReceiver):
    def receive(self) -> str:
        try:
            return input("Введите команду (forward <ms>/left/right/stop/exit): ")
        except EOFError:
            return "exit"

