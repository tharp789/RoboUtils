import serial
import time
import sys

def read_usb(port="/dev/ttyUSB0", baudrate=115200, timeout=1):
    """
    Continuously read data from a USB serial device.
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"✅ Connected to {port} at {baudrate} baud.")
        time.sleep(2)  # Allow time for device to initialize

        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode(errors='ignore').strip()
                if data:
                    print(f"Received: {data}")

    except serial.SerialException as e:
        print(f"❌ Serial error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("🔌 Serial port closed.")

if __name__ == "__main__":
    # Optional CLI args: python3 read_usb_serial.py /dev/ttyACM0 9600
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
    # port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    read_usb(port, baudrate)
