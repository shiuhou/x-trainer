import socket
import time
import math
import re

from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_read import (
    GroupSyncRead,
)
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import (
    PacketHandler,
)
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import (
    PortHandler,
)
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
)


# ============================================================
# SAFETY SETTINGS
# ============================================================

DT = 0.03                  # ~33 Hz
MAX_OFFSET_DEG = 5.0       # Leader J1最多控制Nova偏移 ±5°
MAX_SPEED_DEG_S = 2.0      # Nova target最多以 2 deg/s 改变
LEADER_STALE_TIMEOUT = 0.20

SERVO_T = 0.1
SERVO_AHEADTIME = 50
SERVO_GAIN = 500


# ============================================================
# X-Trainer RIGHT Leader
# ============================================================

LEADER_PORT = (
    "/dev/serial/by-id/"
    "SET_LEADER_RIGHT_SERIAL"
)

LEADER_BAUD = 2_000_000

JOINT_IDS = [11, 12, 14, 15, 16, 17]
APPEND_ID = 13

ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 140
LEN_PRESENT_POSITION = 4

OFFSETS = [
    0.8,
    2.36,
    1.6,
    3.88,
    5.48,
    1.46,
]

SIGNS = [
    1,
    1,
    -1,
    1,
    1,
    1,
]


# ============================================================
# Nova
# ============================================================

NOVA_IP = "SET_NOVA_IP"
NOVA_PORT = 29999


# ============================================================
# Helpers
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def raw_to_signed(raw):
    if raw >= 2**31:
        raw -= 2**32
    return raw


def raw_to_rad(raw):
    return raw_to_signed(raw) / 2048.0 * math.pi


def send_dobot(sock, command):
    sock.sendall(command.encode("ascii"))

    data = b""

    while b";" not in data:
        chunk = sock.recv(4096)

        if not chunk:
            raise RuntimeError(
                "Nova TCP connection closed"
            )

        data += chunk

    return data.decode(
        "ascii",
        errors="replace",
    )


def get_error_id(response):
    """
    Example:
        0,{123},ServoJ(...);
       -1,{},ServoJ(...);

    Return first integer.
    """
    m = re.match(
        r"\s*(-?\d+)\s*,",
        response,
    )

    if not m:
        raise RuntimeError(
            f"Cannot parse ErrorID: {response}"
        )

    return int(m.group(1))


def parse_brace_numbers(response):
    m = re.search(
        r"\{([^}]*)\}",
        response,
    )

    if not m:
        raise RuntimeError(
            f"Cannot parse response: {response}"
        )

    content = m.group(1).strip()

    if not content:
        return []

    return [
        float(x.strip())
        for x in content.split(",")
    ]


# ============================================================
# Leader setup
# ============================================================

print()
print("==========================================")
print(" X-Trainer J1 -> Nova J1 ServoJ TEST")
print("==========================================")
print()
print("Nova motion WILL occur.")
print(f"Maximum offset : +/-{MAX_OFFSET_DEG:.1f} deg")
print(f"Maximum speed  : {MAX_SPEED_DEG_S:.1f} deg/s")
print()

leader_port = PortHandler(
    LEADER_PORT
)

packet = PacketHandler(
    2.0
)

if not leader_port.openPort():
    raise RuntimeError(
        f"Cannot open Leader:\n{LEADER_PORT}"
    )

if not leader_port.setBaudRate(
    LEADER_BAUD
):
    leader_port.closePort()

    raise RuntimeError(
        f"Cannot set baudrate {LEADER_BAUD}"
    )

print("Leader connected.")


# ============================================================
# Unlock Leader
# ============================================================

print("Disabling Leader torque...")

for motor_id in JOINT_IDS + [APPEND_ID]:

    result, error = packet.write1ByteTxRx(
        leader_port,
        motor_id,
        ADDR_TORQUE_ENABLE,
        0,
    )

    if (
        result == COMM_SUCCESS
        and error == 0
    ):
        print(
            f"  ID {motor_id}: torque OFF"
        )
    else:
        leader_port.closePort()

        raise RuntimeError(
            f"Cannot disable torque "
            f"ID={motor_id}: "
            f"result={result}, error={error}"
        )


# ============================================================
# Leader position reader
# ============================================================

reader = GroupSyncRead(
    leader_port,
    packet,
    ADDR_PRESENT_POSITION,
    LEN_PRESENT_POSITION,
)

for motor_id in JOINT_IDS:

    if not reader.addParam(
        motor_id
    ):
        leader_port.closePort()

        raise RuntimeError(
            f"Cannot add Leader ID "
            f"{motor_id}"
        )


def get_leader_deg():

    result = reader.txRxPacket()

    if result != COMM_SUCCESS:
        return None

    positions = []

    for i, motor_id in enumerate(
        JOINT_IDS
    ):

        if not reader.isAvailable(
            motor_id,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        ):
            return None

        raw = reader.getData(
            motor_id,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )

        raw_rad = raw_to_rad(raw)

        calibrated_rad = (
            raw_rad - OFFSETS[i]
        ) * SIGNS[i]

        positions.append(
            math.degrees(
                calibrated_rad
            )
        )

    return positions


# ============================================================
# Connect Nova
# ============================================================

nova = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

nova.settimeout(1.0)

servo_started = False

try:

    print()
    print(
        f"Connecting Nova "
        f"{NOVA_IP}:{NOVA_PORT} ..."
    )

    nova.connect(
        (NOVA_IP, NOVA_PORT)
    )

    print("Nova connected.")


    # ========================================================
    # Safety check: RobotMode must be ENABLE idle = 5
    # ========================================================

    mode_resp = send_dobot(
        nova,
        "RobotMode()",
    )

    print(
        "RobotMode:",
        mode_resp.strip(),
    )

    mode_values = parse_brace_numbers(
        mode_resp
    )

    if (
        len(mode_values) != 1
        or int(mode_values[0]) != 5
    ):
        raise RuntimeError(
            "Nova is NOT in RobotMode 5 "
            "(ENABLE idle).\n"
            "No motion command sent."
        )


    # ========================================================
    # Check alarms
    # ========================================================

    err_resp = send_dobot(
        nova,
        "GetErrorID()",
    )

    print(
        "GetErrorID:",
        err_resp.strip(),
    )


    # ========================================================
    # Read Nova anchor
    # ========================================================

    angle_resp = send_dobot(
        nova,
        "GetAngle()",
    )

    nova_zero = parse_brace_numbers(
        angle_resp
    )

    if len(nova_zero) != 6:
        raise RuntimeError(
            f"Expected 6 Nova angles, "
            f"got {nova_zero}"
        )

    print()
    print(
        "Nova current:",
        " ".join(
            f"{x:7.2f}"
            for x in nova_zero
        ),
    )


    # ========================================================
    # Read Leader anchor
    # ========================================================

    leader_zero = None

    timeout_start = time.monotonic()

    while leader_zero is None:

        leader_zero = get_leader_deg()

        if (
            time.monotonic()
            - timeout_start
            > 2.0
        ):
            raise RuntimeError(
                "Cannot read Leader position."
            )

        time.sleep(0.01)

    print(
        "Leader current:",
        " ".join(
            f"{x:7.2f}"
            for x in leader_zero
        ),
    )


    # ========================================================
    # ARM
    # ========================================================

    print()
    print(
        "------------------------------------------"
    )
    print("PRE-FLIGHT:")
    print("  - Nova周围无人/无障碍")
    print("  - 急停按钮可立即按到")
    print("  - 只会控制 Nova J1")
    print(
        f"  - 最大偏移 +/-"
        f"{MAX_OFFSET_DEG} deg"
    )
    print(
        f"  - 最大速度 "
        f"{MAX_SPEED_DEG_S} deg/s"
    )
    print()
    print(
        "保持 Leader 当前姿态。"
    )
    print(
        "按 Enter 开始真实 ServoJ 跟随。"
    )
    print(
        "Ctrl+C 随时停止。"
    )
    print(
        "------------------------------------------"
    )

    input()


    # Re-anchor immediately before motion

    leader_zero = get_leader_deg()

    if leader_zero is None:
        raise RuntimeError(
            "Leader read failed before start."
        )

    angle_resp = send_dobot(
        nova,
        "GetAngle()",
    )

    nova_zero = parse_brace_numbers(
        angle_resp
    )

    if len(nova_zero) != 6:
        raise RuntimeError(
            "Failed to re-read Nova anchor."
        )


    # Current commanded target begins at actual Nova angle

    commanded = nova_zero.copy()

    last_leader_ok = time.monotonic()
    last_loop = time.monotonic()

    servo_started = True

    print()
    print("SERVO ACTIVE")
    print("Move Leader J1 slowly.")
    print()


    # ========================================================
    # Servo loop
    # ========================================================

    while True:

        loop_start = time.monotonic()

        leader_now = get_leader_deg()


        # ----------------------------------------------------
        # Leader watchdog
        # ----------------------------------------------------

        if leader_now is None:

            if (
                time.monotonic()
                - last_leader_ok
                > LEADER_STALE_TIMEOUT
            ):
                print()
                print(
                    "Leader data stale > "
                    f"{LEADER_STALE_TIMEOUT}s"
                )
                print(
                    "Stopping Nova."
                )

                try:
                    print(
                        send_dobot(
                            nova,
                            "Stop()",
                        ).strip()
                    )
                except Exception:
                    pass

                break

            # Do NOT send a new target
            continue


        last_leader_ok = (
            time.monotonic()
        )


        # ----------------------------------------------------
        # Leader J1 relative movement
        # ----------------------------------------------------

        leader_delta_j1 = (
            leader_now[0]
            - leader_zero[0]
        )

        leader_delta_j1 = clamp(
            leader_delta_j1,
            -MAX_OFFSET_DEG,
            MAX_OFFSET_DEG,
        )


        # Desired Nova J1

        desired_j1 = (
            nova_zero[0]
            + leader_delta_j1
        )


        # ----------------------------------------------------
        # Rate limiter
        # ----------------------------------------------------

        now = time.monotonic()

        actual_dt = max(
            now - last_loop,
            0.001,
        )

        last_loop = now

        max_step = (
            MAX_SPEED_DEG_S
            * actual_dt
        )

        error_j1 = (
            desired_j1
            - commanded[0]
        )

        step_j1 = clamp(
            error_j1,
            -max_step,
            max_step,
        )

        commanded[0] += step_j1


        # J2~J6 remain at original anchor

        commanded[1:] = nova_zero[1:]


        # ----------------------------------------------------
        # ServoJ command
        # ----------------------------------------------------

        cmd = (
            "ServoJ("
            f"{commanded[0]:.6f},"
            f"{commanded[1]:.6f},"
            f"{commanded[2]:.6f},"
            f"{commanded[3]:.6f},"
            f"{commanded[4]:.6f},"
            f"{commanded[5]:.6f},"
            f"t={SERVO_T},"
            f"aheadtime={SERVO_AHEADTIME},"
            f"gain={SERVO_GAIN}"
            ")"
        )

        response = send_dobot(
            nova,
            cmd,
        )

        error_id = get_error_id(
            response
        )


        # ----------------------------------------------------
        # Stop immediately on rejected ServoJ
        # ----------------------------------------------------

        if error_id != 0:

            print()
            print(
                "ServoJ rejected:"
            )
            print(response)

            try:
                print(
                    "RobotMode:",
                    send_dobot(
                        nova,
                        "RobotMode()",
                    ).strip(),
                )

                print(
                    "GetErrorID:",
                    send_dobot(
                        nova,
                        "GetErrorID()",
                    ).strip(),
                )

            except Exception:
                pass

            break


        print(
            "\r"
            f"Leader ΔJ1="
            f"{leader_delta_j1:+6.2f}°  "
            f"desired="
            f"{desired_j1:8.2f}°  "
            f"commanded="
            f"{commanded[0]:8.2f}°",
            end="",
            flush=True,
        )


        # ----------------------------------------------------
        # Maintain ~33 Hz
        # ----------------------------------------------------

        elapsed = (
            time.monotonic()
            - loop_start
        )

        sleep_time = (
            DT - elapsed
        )

        if sleep_time > 0:
            time.sleep(
                sleep_time
            )


except KeyboardInterrupt:

    print()
    print("Ctrl+C received.")


except Exception as e:

    print()
    print("ERROR:", e)


finally:

    # ========================================================
    # Stop Nova servo motion
    # ========================================================

    if servo_started:

        try:
            print()
            print(
                "Sending Stop() ..."
            )

            resp = send_dobot(
                nova,
                "Stop()",
            )

            print(resp.strip())

        except Exception as e:

            print(
                "Stop() failed:",
                e,
            )


    # ========================================================
    # Final diagnostics
    # ========================================================

    try:

        print(
            "Final RobotMode:",
            send_dobot(
                nova,
                "RobotMode()",
            ).strip(),
        )

        print(
            "Final GetErrorID:",
            send_dobot(
                nova,
                "GetErrorID()",
            ).strip(),
        )

    except Exception:
        pass


    try:
        nova.close()
    except Exception:
        pass

    try:
        leader_port.closePort()
    except Exception:
        pass

    print(
        "Connections closed."
    )
