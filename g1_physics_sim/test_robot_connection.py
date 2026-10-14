from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from typing import List, Optional

try:
    from .g1_constants import (
        JOINT_NAMES_29DOF, NUM_BODY_MOTORS, LOWCMD_SLOTS,
        TOPIC_LOW_STATE, DDS_DOMAIN_REAL, MODE_PR, MODE_AB,
    )
except ImportError:
    from g1_constants import (
        JOINT_NAMES_29DOF, NUM_BODY_MOTORS, LOWCMD_SLOTS,
        TOPIC_LOW_STATE, DDS_DOMAIN_REAL, MODE_PR, MODE_AB,
    )

ROBOT_IP = '192.168.123.99'
ROBOT_SUBNET = '192.168.123.'
NETMASK = '255.255.255.0'


def list_local_network_interfaces() -> bool:
    print('=' * 68)
    print('LOCAL NETWORK INTERFACES')
    print('=' * 68)
    found = False
    system = platform.system()
    try:
        if system == 'Windows':
            out = subprocess.check_output('ipconfig', text=True, stderr=subprocess.DEVNULL)
            adapter = ''
            for raw in out.splitlines():
                line = raw.strip()
                if line.endswith('adapter:') or (line.endswith(':') and 'adapter' in line.lower()):
                    adapter = line.rstrip(':').replace(' adapter', '')
                elif 'IPv4 Address' in line or 'IPv4-adresse' in line:
                    ip = line.split(':')[-1].strip().replace('(Preferred)', '')
                    if not ip:
                        continue
                    mark = ''
                    if ip.startswith(ROBOT_SUBNET):
                        mark = '   <-- G1 SUBNET'
                        found = True
                        if ip == ROBOT_IP:
                            mark += '  [!! CONFLICT: this is the robot IP]'
                    print(f'  {adapter:<32} {ip}{mark}')
        else:
            out = subprocess.check_output(['ip', '-br', 'addr'], text=True, stderr=subprocess.DEVNULL)
            for raw in out.splitlines():
                parts = raw.split()
                if len(parts) < 3:
                    continue
                iface, state, cidr = parts[0], parts[1], parts[2]
                ip = cidr.split('/')[0]
                mark = ''
                if ip.startswith(ROBOT_SUBNET):
                    mark = '   <-- G1 SUBNET'
                    found = True
                    if ip == ROBOT_IP:
                        mark += '  [!! CONFLICT: this is the robot IP]'
                print(f'  {iface:<20} [{state}] {ip}{mark}')
    except Exception as e:
        print(f'  (could not enumerate interfaces: {e})')

    if not found:
        print(f'\n  No adapter on {ROBOT_SUBNET}x. Configure the Ethernet NIC:')
        print(f'      address : {ROBOT_SUBNET}222   (NOT {ROBOT_IP})')
        print(f'      netmask : {NETMASK}')
    return found


def check_robot_dds(iface: str, domain_id: int = DDS_DOMAIN_REAL,
                    timeout: float = 5.0) -> bool:
    print()
    print('=' * 68)
    print(f"DDS CHECK  iface='{iface}'  domain={domain_id}  topic='{TOPIC_LOW_STATE}'")
    print('=' * 68)
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as e:
        print(f'[ERROR] unitree_sdk2py not importable: {e}')
        print('        git clone https://github.com/unitreerobotics/unitree_sdk2_python.git')
        print('        cd unitree_sdk2_python && pip install -e .')
        return False

    received: List[bool] = [False]
    last: List[Optional[object]] = [None]
    crc_bad = [0]
    crc = CRC()

    def on_lowstate(msg):
        try:
            if int(crc.Crc(msg)) != int(getattr(msg, 'crc', 0)):
                crc_bad[0] += 1
                return
        except Exception:
            pass
        received[0] = True
        last[0] = msg

    try:
        ChannelFactoryInitialize(domain_id, iface)
        sub = ChannelSubscriber(TOPIC_LOW_STATE, LowState_)
        sub.Init(on_lowstate, 10)
    except Exception as e:
        print(f'[ERROR] CycloneDDS init failed on {iface!r}: {e}')
        return False

    print(f'  listening up to {timeout:.0f}s ...')
    start = time.time()
    while time.time() - start < timeout and not received[0]:
        time.sleep(0.05)

    if not received[0]:
        print('\n[!] No valid LowState received.')
        print('    Checklist:')
        print(f'      1. Robot powered on and in developer mode (hold R2+L2 for 20 s).')
        print(f'      2. Ethernet cable seated at both ends.')
        print(f'      3. PC adapter set to {ROBOT_SUBNET}222/{NETMASK} (not {ROBOT_IP}).')
        print(f'      4. --iface matches the NIC name exactly (case-sensitive).')
        print(f'      5. Domain {domain_id} matches what the robot expects (usually 0).')
        print(f'      6. ping {ROBOT_IP} responds.')
        return False

    msg = last[0]
    n = len(msg.motor_state)
    print(f'\n[OK] LowState stream is live and CRC-valid.')
    print(f'     motor_state slots : {n}   (LowCmd_ has {LOWCMD_SLOTS})')
    if crc_bad[0]:
        print(f'     dropped by CRC   : {crc_bad[0]}')

    mm = int(getattr(msg, 'mode_machine', -1))
    pr = int(getattr(msg, 'mode_pr', -1))
    print(f'     mode_machine     : {mm}  (robot variant; must be non-zero once reported)')
    print(f'     mode_pr          : {pr}  ({"PR/series" if pr == MODE_PR else "AB/parallel"})')

    if n < NUM_BODY_MOTORS:
        print(f'\n[ERROR] Only {n} motor slots; a 29-DOF G1 must report '
              f'{NUM_BODY_MOTORS}. Is this the right robot?')
        return False

    print('\n--- joints ---')
    print(f"{'idx':<5}{'joint':<28}{'pos(deg)':>10}{'vel(rad/s)':>11}"
          f"{'tau(Nm)':>9}{'temp(C)':>9}{'code':>6}")
    print('-' * 78)
    faulted: List[int] = []
    for i in range(NUM_BODY_MOTORS):
        ms = msg.motor_state[i]
        code = int(getattr(ms, 'motorstate', 0))
        if code:
            faulted.append(i)
        temps = getattr(ms, 'temperature', None)
        if temps is None:
            temp = float('nan')
        else:
            temp = float(max(temps))
        print(f'{i:<5}{JOINT_NAMES_29DOF[i]:<28}{float(ms.q) * 57.29578:>10.2f}'
              f'{float(ms.dq):>11.3f}{float(ms.tau_est):>9.2f}{temp:>9.1f}{code:>6}')

    if faulted:
        print(f'\n[ERROR] motor fault codes on indices {faulted}. Do not command this robot.')
        return False
    if mm == 0:
        print('\n[WARN] mode_machine still 0; LowState may be partial. '
              'The bridge refuses to stream on this value.')

    print('\n[READY] Frame contract verified: 29 slots, CRC clean, no motor faults.')
    print('         Next: python run_real.py --check-only --iface '
          f'{iface} --domain {domain_id}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description='Unitree G1 connectivity diagnostics')
    ap.add_argument('--iface', default='Ethernet',
                    help='NIC name, case-sensitive (eth0, enp2s0, Ethernet)')
    ap.add_argument('--domain', type=int, default=DDS_DOMAIN_REAL)
    ap.add_argument('--timeout', type=float, default=5.0)
    ap.add_argument('--net-only', action='store_true')
    args = ap.parse_args()

    have_subnet = list_local_network_interfaces()
    if args.net_only:
        return 0 if have_subnet else 1

    ok = check_robot_dds(args.iface, args.domain, args.timeout)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())