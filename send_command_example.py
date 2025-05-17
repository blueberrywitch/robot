import sys, time
import paho.mqtt.client as mqtt

def main() -> None:
    payload = "forward 1000" if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    client  = mqtt.Client()
    client.connect("localhost", 1883, 60)
    client.loop_start()
    print(f"MQTT >> {payload}")
    client.publish("/motor/command", payload)
    time.sleep(0.5)
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()

