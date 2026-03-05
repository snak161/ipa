import serial
import serial.tools.list_ports
import threading
import datetime

BAUDRATE = 9600
APPEND_CR = True


def timestamp():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def hexdump(data):
    return " ".join(f"{b:02X}" for b in data)


def choose_port():
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("Keine seriellen Ports gefunden.")
        exit()

    print("Verfügbare Ports:\n")

    for i, p in enumerate(ports):
        print(f"{i+1}: {p.device} - {p.description}")

    print()

    while True:
        try:
            choice = int(input("Port wählen: ")) - 1
            if 0 <= choice < len(ports):
                return ports[choice].device
        except:
            pass

        print("Ungültige Auswahl.")


def reader(ser):
    while True:
        try:
            data = ser.readline()
            if data:
                text = data.decode(errors="replace").rstrip()
                print(f"\n[{timestamp()}] << {text} | HEX: {hexdump(data)}")
                print(">> ", end="", flush=True)
        except:
            break


port = choose_port()

ser = serial.Serial(port, BAUDRATE, timeout=0.1)

print(f"\nConnected to {port} @ {BAUDRATE}")
print("Type commands. 'exit' to quit.\n")

thread = threading.Thread(target=reader, args=(ser,), daemon=True)
thread.start()

while True:

    cmd = input(">> ")

    if cmd.lower() in ["exit", "quit"]:
        break

    data = cmd.encode()

    if APPEND_CR:
        data += b"\r"

    ser.write(data)

    print(f"[{timestamp()}] >> {cmd} | HEX: {hexdump(data)}")

ser.close()
print("Connection closed")