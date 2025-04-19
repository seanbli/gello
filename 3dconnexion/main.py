import hid
import struct
import time

# 查找 SpaceMouse 设备
spacemouse = None
for device in hid.enumerate():
    if device['vendor_id'] == 0x46d:  # 3Dconnexion
        print("Found:", device['product_string'])
        spacemouse = device
        break

if not spacemouse:
    print("SpaceMouse not found.")
    exit()

# 打开设备
device = hid.device()
device.open_path(spacemouse['path'])
device.set_nonblocking(True)

print("Reading SpaceMouse input... Press Ctrl+C to exit.")

try:
    while True:
        data = device.read(13)
        if not data:
            time.sleep(0.01)
            continue

        report_id = data[0]

        if report_id == 1:
            # Translation
            tx = struct.unpack('<h', bytes(data[1:3]))[0]
            ty = struct.unpack('<h', bytes(data[3:5]))[0]
            tz = struct.unpack('<h', bytes(data[5:7]))[0]
            print(f"[Translation] x={tx}, y={ty}, z={tz}")

        elif report_id == 2:
            # Rotation
            rx = struct.unpack('<h', bytes(data[1:3]))[0]
            ry = struct.unpack('<h', bytes(data[3:5]))[0]
            rz = struct.unpack('<h', bytes(data[5:7]))[0]
            print(f"[Rotation] rx={rx}, ry={ry}, rz={rz}")

        elif report_id == 3:
            # Buttons (bitmask: first byte)
            buttons = data[1]
            left_pressed = bool(buttons & 0x01)
            right_pressed = bool(buttons & 0x02)
            print(f"[Buttons] Left: {'Pressed' if left_pressed else 'Released'}, Right: {'Pressed' if right_pressed else 'Released'}")

except KeyboardInterrupt:
    print("\nExiting.")
    device.close()
