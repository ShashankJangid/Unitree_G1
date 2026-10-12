import pyrealsense2 as rs

def main() -> None:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print('No RealSense devices found.')
        return
    print(f'Found {len(devices)} RealSense device(s):\n')
    for idx, dev in enumerate(devices):
        print(f'[{idx}] {dev.get_info(rs.camera_info.name)}')
        print(f'    Serial number : {dev.get_info(rs.camera_info.serial_number)}')
        print(f'    Firmware ver. : {dev.get_info(rs.camera_info.firmware_version)}')
        if dev.supports(rs.camera_info.physical_port):
            print(f'    USB port      : {dev.get_info(rs.camera_info.physical_port)}')
        print()
if __name__ == '__main__':
    main()
