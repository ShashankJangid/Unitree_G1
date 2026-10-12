from __future__ import annotations
import argparse
import time
import curses
try:
    from pynput.keyboard import Listener, Key, KeyCode
except ModuleNotFoundError as exc:
    raise SystemExit("The 'pynput' package is required for keyboard_controller.py.\nInstall with:  pip install pynput") from exc
from hanger_boot_sequence import hanger_boot_sequence
LIN_STEP = 0.05
ANG_STEP = 0.2
SEND_PERIOD = 0.1

def clamp(value: float, limit: float=0.6) -> float:
    return max(-limit, min(limit, value))

def drive_loop(stdscr: 'curses._CursesWindow', bot) -> None:
    curses.cbreak()
    stdscr.nodelay(True)
    vx = vy = omega = 0.0
    last_send = 0.0
    pressed_keys: set[object] = set()

    def _on_press(key):
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.add(key.char.lower())
        else:
            pressed_keys.add(key)

    def _on_release(key):
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.discard(key.char.lower())
        else:
            pressed_keys.discard(key)
    listener = Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    def key(name: str) -> bool:
        if name == 'space':
            return Key.space in pressed_keys
        if name == 'esc':
            return Key.esc in pressed_keys
        return name in pressed_keys
    try:
        while True:
            if key('w') and (not key('s')):
                vx = clamp(vx + LIN_STEP)
            elif key('s') and (not key('w')):
                vx = clamp(vx - LIN_STEP)
            else:
                vx = 0.0
            if key('q') and (not key('e')):
                vy = clamp(vy + LIN_STEP)
            elif key('e') and (not key('q')):
                vy = clamp(vy - LIN_STEP)
            else:
                vy = 0.0
            if key('a') and (not key('d')):
                omega = clamp(omega + ANG_STEP)
            elif key('d') and (not key('a')):
                omega = clamp(omega - ANG_STEP)
            else:
                omega = 0.0
            if key('space'):
                vx = vy = omega = 0.0
            if key('z'):
                bot.Damp()
                break
            if key('esc'):
                bot.StopMove()
                bot.ZeroTorque()
                break
            now = time.time()
            if now - last_send >= SEND_PERIOD:
                bot.Move(vx, vy, omega, continous_move=True)
                last_send = now
                stdscr.erase()
                stdscr.addstr(0, 0, 'Hold keys to drive – Z: quit  ESC: e-stop')
                stdscr.addstr(2, 0, f'vx: {vx:+.2f}  vy: {vy:+.2f}  omega: {omega:+.2f}')
                stdscr.refresh()
            time.sleep(0.005)
    finally:
        listener.stop()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iface', default='enp68s0f1', help='network interface connected to robot')
    args = parser.parse_args()
    bot = hanger_boot_sequence(iface=args.iface)
    curses.wrapper(drive_loop, bot)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted – sending Damp …')
        try:
            bot.Damp()
        except Exception:
            pass
