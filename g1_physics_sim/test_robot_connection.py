import argparse
import sys
import time
import socket
import numpy as np
from g1_constants import JOINT_NAMES_29DOF, NUM_MOTORS

def list_local_network_interfaces():
    print('=' * 65)
    print('LOCAL NETWORK INTERFACES DIAGNOSTIC')
    print('=' * 65)
    import subprocess
    import platform
    found_g1_subnet = False
    system = platform.system()
    if system == 'Windows':
        try:
            out = subprocess.check_output('ipconfig', text=True, stderr=subprocess.DEVNULL)
            current_adapter = ''
            for line in out.splitlines():
                line = line.strip()
                if 'adapter' in line.lower() and line.endswith(':'):
                    current_adapter = line.rstrip(':')
                elif 'IPv4 Address' in line or 'IP Address' in line:
                    ip = line.split(':')[-1].strip().replace('(Preferred)', '')
                    mark = ''
                    if ip.startswith('192.168.123.'):
                        mark = '  <-- [MATCHES G1 SUBNET 192.168.123.x!]'
                        found_g1_subnet = True
                    print(f'  • {current_adapter:<30} IP: {ip}{mark}')
        except Exception:
            pass
    else:
        try:
            out = subprocess.check_output(['ip', '-br', 'addr'], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    iface, state, ip_cidr = (parts[0], parts[1], parts[2])
                    ip = ip_cidr.split('/')[0]
                    mark = ''
                    if ip.startswith('192.168.123.'):
                        mark = '  <-- [MATCHES G1 SUBNET 192.168.123.x!]'
                        found_g1_subnet = True
                    print(f'  • {iface:<20} [{state}]: {ip}{mark}')
        except Exception:
            pass
    if not found_g1_subnet:
        print('\n  [!] Notice: No interface currently has an IP on 192.168.123.x.')
        print('      To connect to G1, set your Ethernet adapter to static IP 192.168.123.99 / 255.255.255.0.')

def check_robot_dds(iface: str, domain_id: int=0, timeout: float=3.0):
    print('\n' + '=' * 65)
    print(f"CHECKING UNITREE G1 DDS COMMUNICATION (Interface: '{iface}')")
    print('=' * 65)
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except ImportError as e:
        print(f'\n[ERROR] unitree_sdk2py is not installed or importable: {e}')
        print('Install via: cd unitree_sdk2_python && pip install -e .')
        return False
    received_state = [False]
    last_msg = [None]

    def on_lowstate(msg):
        received_state[0] = True
        last_msg[0] = msg
    try:
        ChannelFactoryInitialize(domain_id, iface)
        sub = ChannelSubscriber('rt/lowstate', LowState_)
        sub.Init(on_lowstate, 10)
        print(f"Listening for G1 telemetry on 'rt/lowstate' (Timeout: {timeout}s)...")
    except Exception as e:
        print(f"[ERROR] Failed to initialize CycloneDDS on '{iface}': {e}")
        return False
    start = time.time()
    while time.time() - start < timeout:
        if received_state[0]:
            break
        time.sleep(0.05)
    if not received_state[0]:
        print('\n[!] No DDS response from G1 robot.')
        print('    Troubleshooting checklist:')
        print('    1. Is the G1 robot powered on and connected via Ethernet cable?')
        print('    2. Is your PC Ethernet configured to static IP 192.168.123.99 / 255.255.255.0?')
        print('    3. Can you ping the robot: ping 192.168.123.161?')
        return False
    msg = last_msg[0]
    print('\n[SUCCESS] Connected to Unitree G1 EDU robot over DDS!')
    print(f'Total motors reported: {len(msg.motor_state)}')
    print('\n--- Live Arm Joint Feedback ---')
    print(f"{'Index':<6} {'Joint Name':<28} {'Pos (deg)':<12} {'Vel (rad/s)':<12} {'Temp (C)':<10}")
    print('-' * 68)
    for i in range(15, min(NUM_MOTORS, len(msg.motor_state))):
        ms = msg.motor_state[i]
        deg = float(ms.q) * (180.0 / 3.14159)
        vel = float(ms.dq)
        temp = getattr(ms, 'temperature', 0)
        print(f'{i:<6} {JOINT_NAMES_29DOF[i]:<28} {deg:>8.2f}° {vel:>10.2f} {temp:>8}')
    print('\nRobot is ready for control streaming.')
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Unitree G1 robot connection and diagnostics')
    parser.add_argument('--iface', default='Ethernet', help='Network interface name (e.g. Ethernet, eth0)')
    parser.add_argument('--domain', type=int, default=0, help='DDS domain ID')
    args = parser.parse_args()
    list_local_network_interfaces()
    check_robot_dds(args.iface, args.domain)
