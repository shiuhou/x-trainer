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
# Leader
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
# Mapping
#
# 暂时全部 1:1。
# 如果之后发现某轴方向反了，把对应 1 改成 -1。
# ============================================================

MAP_SIGN = [
    1,
    1,
    1,
    1,
    1,
    1,
]

SCALE = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]


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
            raise RuntimeError("Nova TCP connection closed")

        data += chunk

    return data.decode("ascii", errors="replace")


def parse_brace_numbers(response):
    match = re.search(r"\{([^}]*)\}", response)

    if not match:
        raise RuntimeError(
            f"Cannot parse response:\n{response}"
        )

    text = match.group(1)

    return [
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


# ============================================================
# Open Leader
# ============================================================

print("=== Leader -> Nova DRY RUN ===")
print("NO Nova motion commands will be sent.\n")

leader_port = PortHandler(LEADER_PORT)
packet = PacketHandler(2.0)

if not leader_port.openPort():
    raise RuntimeError(
        f"Cannot open Leader: {LEADER_PORT}"
    )

if not leader_port.setBaudRate(LEADER_BAUD):
    leader_port.closePort()
    raise RuntimeError(
        f"Cannot set Leader baud: {LEADER_BAUD}"
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

    if result == COMM_SUCCESS and error == 0:
        print(f"  ID {motor_id}: torque OFF")
    else:
        print(
            f"  ID {motor_id}: "
            f"result={result}, error={error}"
        )


# ============================================================
# Leader reader
# ============================================================

reader = GroupSyncRead(
    leader_port,
    packet,
    ADDR_PRESENT_POSITION,
    LEN_PRESENT_POSITION,
)

for motor_id in JOINT_IDS:
    if not reader.addParam(motor_id):
        raise RuntimeError(
            f"Cannot add Leader ID {motor_id}"
        )


def get_leader_deg():

    result = reader.txRxPacket()

    if result != COMM_SUCCESS:
        return None

    result_deg = []

    for i, motor_id in enumerate(JOINT_IDS):

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

        result_deg.append(
            math.degrees(calibrated_rad)
        )

    return result_deg


# ============================================================
# Connect Nova
# ============================================================

nova = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

nova.settimeout(2.0)

print()
print(
    f"Connecting Nova "
    f"{NOVA_IP}:{NOVA_PORT} ..."
)

nova.connect(
    (NOVA_IP, NOVA_PORT)
)

print("Nova connected.")

mode_resp = send_dobot(
    nova,
    "RobotMode()",
)

angle_resp = send_dobot(
    nova,
    "GetAngle()",
)

print("RobotMode:", mode_resp.strip())
print("GetAngle :", angle_resp.strip())

nova_angles = parse_brace_numbers(
    angle_resp
)

if len(nova_angles) != 6:
    raise RuntimeError(
        f"Expected 6 Nova joints, got "
        f"{nova_angles}"
    )


# ============================================================
# Anchor
# ============================================================

print()
print("--------------------------------------------------")
print("把 Leader 和 Nova 放在你想作为对应零点的位置。")
print()
print("Nova 不会移动。")
print("准备好后按 Enter 记录 anchor。")
print("--------------------------------------------------")

input()

leader_zero = None

while leader_zero is None:
    leader_zero = get_leader_deg()
    time.sleep(0.02)

nova_zero = nova_angles.copy()

print()
print("ANCHOR RECORDED")
print()

print(
    "Leader0:",
    " ".join(
        f"{x:7.2f}"
        for x in leader_zero
    )
)

print(
    "Nova0  :",
    " ".join(
        f"{x:7.2f}"
        for x in nova_zero
    )
)

print()
print("Move the Leader.")
print("Nova WILL NOT MOVE.")
print("Ctrl+C to stop.")
print()


# ============================================================
# Dry-run
# ============================================================

try:

    while True:

        leader_now = get_leader_deg()

        if leader_now is None:
            print(
                "\rLeader read failed",
                end="",
                flush=True,
            )

            time.sleep(0.05)
            continue


        leader_delta = [
            leader_now[i] - leader_zero[i]
            for i in range(6)
        ]


        nova_target = [
            nova_zero[i]
            + leader_delta[i]
            * MAP_SIGN[i]
            * SCALE[i]

            for i in range(6)
        ]


        print(
            "\r"
            "ΔLeader [deg]: "
            + " ".join(
                f"{x:7.2f}"
                for x in leader_delta
            )
            + "   |   "
            "Nova target: "
            + " ".join(
                f"{x:7.2f}"
                for x in nova_target
            ),
            end="",
            flush=True,
        )


        time.sleep(0.05)


except KeyboardInterrupt:
    print("\nStopped.")


finally:

    nova.close()
    leader_port.closePort()

    print("Connections closed.")

