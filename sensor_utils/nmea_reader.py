import serial
import pynmea2
import sys
import time

def read_nmea(port="/dev/ttyUSB0", baudrate=115200, timeout=1):
    """
    Reads NMEA sentences from a GPS device over serial and parses them.
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"✅ Connected to {port} at {baudrate} baud.")
        time.sleep(2)  # Allow time for device to initialize

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors='ignore').strip()
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)
                        
                        if isinstance(msg, pynmea2.types.talker.GGA):
                            print(f"🛰  GGA - Fix Quality: {msg.gps_qual}, "
                                f"Satellites: {msg.num_sats}, "
                                f"Altitude: {msg.altitude} {msg.altitude_units}")
                            print(f"📍 Position: {msg.latitude:.6f}, {msg.longitude:.6f}")

                        elif isinstance(msg, pynmea2.types.talker.RMC):
                            print(f"🕒 RMC - Time: {msg.datestamp} {msg.timestamp}, "
                                f"Speed: {msg.spd_over_grnd} knots")
                            print(f"📍 Position: {msg.latitude:.6f}, {msg.longitude:.6f}")

                    except pynmea2.nmea.ParseError:
                        # Ignore malformed lines
                        continue

    except serial.SerialException as e:
        print(f"❌ Serial error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("🔌 Serial port closed.")

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else 9600
    read_nmea(port, baudrate)