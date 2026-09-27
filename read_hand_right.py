import time
import math

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
# X-Trainer RIGHT Leader configuration
# ============================================================

PORT = "/dev/serial/by-id/SET_LEADER_RIGHT_SERIAL"

BAUD = 2_000_000

# 6 leader joints
JOINT_IDS = [11, 12, 14, 15, 16, 17]

# Additional motor used by X-Trainer
APPEND_ID = 13

# Trigger / gripper motor
TRIGGER_ID = 18

# What we actually read
READ_IDS = JOINT_IDS + [TRIGGER_ID]


# ============================================================
# Dynamixel registers
# Taken from the X-Trainer implementation
# ============================================================

ADDR_TORQUE_ENABLE = 64

ADDR_PRESENT_POSITION = 140
LEN_PRESENT_POSITION = 4


# ============================================================
# Calibration from dobot_settings.ini
# ============================================================

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


def raw_to_signed(raw):
    """
    Convert unsigned 32-bit Dynamixel value into signed int32.
    """
    if raw >= 2**31:
        raw -= 2**32

    return raw


def raw_to_rad(raw):
    """
    X-Trainer conversion:
        raw / 2048 * pi
    """
    return raw_to_signed(raw) / 2048.0 * math.pi


# ============================================================
# Open serial port
# ============================================================

print("Opening X-Trainer Leader...")

port = PortHandler(PORT)
packet = PacketHandler(2.0)

if not port.openPort():
    raise RuntimeError(
        f"Failed to open port:\n{PORT}"
    )

print(f"Port opened: {PORT}")

if not port.setBaudRate(BAUD):
    port.closePort()

    raise RuntimeError(
        f"Failed to set baudrate to {BAUD}"
    )

print(f"Baudrate: {BAUD}")


# ============================================================
# Disable Leader torque
# ============================================================

print()
print("Disabling Leader torque...")

TORQUE_IDS = JOINT_IDS + [APPEND_ID, TRIGGER_ID]

for motor_id in TORQUE_IDS:

    try:
        result, error = packet.write1ByteTxRx(
            port,
            motor_id,
            ADDR_TORQUE_ENABLE,
            0,
        )

        if result == COMM_SUCCESS and error == 0:
            print(
                f"  ID {motor_id:2d}: torque OFF OK"
            )

        else:
            print(
                f"  ID {motor_id:2d}: "
                f"result={result}, error={error}"
            )

    except Exception as e:
        print(
            f"  ID {motor_id:2d}: "
            f"exception: {e}"
        )


print()
print("Leader torque disable command completed.")
print("Try moving the arm by hand now.")
print()


# ============================================================
# Setup GroupSyncRead
# ============================================================

reader = GroupSyncRead(
    port,
    packet,
    ADDR_PRESENT_POSITION,
    LEN_PRESENT_POSITION,
)

for motor_id in READ_IDS:

    if not reader.addParam(motor_id):

        port.closePort()

        raise RuntimeError(
            f"Failed to add motor ID {motor_id} "
            f"to GroupSyncRead"
        )


print("Position reader ready.")
print()
print("Move ONE joint at a time.")
print("Press Ctrl+C to stop.")
print()


# ============================================================
# Main read loop
# ============================================================

try:

    while True:

        result = reader.txRxPacket()

        if result != COMM_SUCCESS:

            print(
                f"\rCommunication error: {result}",
                end="",
                flush=True,
            )

            time.sleep(0.1)
            continue


        raw_radians = []

        valid = True

        for motor_id in READ_IDS:

            available = reader.isAvailable(
                motor_id,
                ADDR_PRESENT_POSITION,
                LEN_PRESENT_POSITION,
            )

            if not available:

                print(
                    f"\nID {motor_id}: "
                    f"position data unavailable"
                )

                valid = False
                break


            raw = reader.getData(
                motor_id,
                ADDR_PRESENT_POSITION,
                LEN_PRESENT_POSITION,
            )

            angle_rad = raw_to_rad(raw)

            raw_radians.append(angle_rad)


        if not valid:
            time.sleep(0.1)
            continue


        # ====================================================
        # Apply X-Trainer calibration
        # q = (raw - offset) * sign
        # ====================================================

        joints = []

        for i in range(6):

            q = (
                raw_radians[i]
                - OFFSETS[i]
            ) * SIGNS[i]

            joints.append(q)


        trigger_rad = raw_radians[6]


        # ====================================================
        # Print in degrees
        # ====================================================

        joint_deg = [
            math.degrees(q)
            for q in joints
        ]

        trigger_deg = math.degrees(
            trigger_rad
        )


        print(
            "\r"
            f"J1={joint_deg[0]:8.2f}°  "
            f"J2={joint_deg[1]:8.2f}°  "
            f"J3={joint_deg[2]:8.2f}°  "
            f"J4={joint_deg[3]:8.2f}°  "
            f"J5={joint_deg[4]:8.2f}°  "
            f"J6={joint_deg[5]:8.2f}°  "
            f"trigger={trigger_deg:8.2f}°",
            end="",
            flush=True,
        )


        # ~20 Hz display
        time.sleep(0.05)


except KeyboardInterrupt:

    print()
    print("Ctrl+C received.")


finally:

    port.closePort()

    print("Port closed.")
