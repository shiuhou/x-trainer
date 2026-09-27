import socket
import time
import math
import re
import json
import os
import queue
import statistics
import threading

from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_read import GroupSyncRead
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import PacketHandler
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import PortHandler
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.robotis_def import COMM_SUCCESS


# ============================================================
# SAFETY / SERVO SETTINGS
# ============================================================

DT = 0.03                      # ~33 Hz

MAX_OFFSET_DEG = [
    20.0, 20.0, 20.0,
    20.0, 20.0, 20.0,
]

MAX_SPEED_DEG_S = [
    15.0, 15.0, 15.0,
    15.0, 15.0, 15.0,
]

LEADER_STALE_TIMEOUT = 0.20

SERVO_T = 0.1
SERVO_AHEADTIME = 50
SERVO_GAIN = 500

TIMING_SUMMARY_PERIOD_S = 2.0
DISPLAY_PERIOD_S = 0.1
TRACE_QUEUE_SIZE = 512
TRACE_FLUSH_PERIOD_S = 0.5
TRACE_DIR = "teleop_traces"
TRACE_SCHEMA_VERSION = 1


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
# Leader -> Nova mapping
# 已经 dry-run 验证为 1:1
# ============================================================

MAP_SIGN = [
    1,-1,-1,
    1,-1,1,
]

SCALE = [
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
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


def send_dobot(sock, command, timing=None):
    if timing is not None:
        timing["t_servoj_send"] = time.monotonic_ns()

    sock.sendall(command.encode("ascii"))

    data = b""

    while b";" not in data:
        chunk = sock.recv(4096)

        if not chunk:
            raise RuntimeError(
                "Nova TCP connection closed"
            )

        data += chunk

    if timing is not None:
        timing["t_servoj_response"] = time.monotonic_ns()

    return data.decode(
        "ascii",
        errors="replace",
    )


def parse_brace_numbers(response):
    m = re.search(r"\{([^}]*)\}", response)

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


def parse_error_id(response):
    m = re.match(
        r"\s*(-?\d+)\s*,",
        response,
    )

    if not m:
        raise RuntimeError(
            f"Cannot parse ErrorID: {response}"
        )

    return int(m.group(1))


def make_trace_path():
    os.makedirs(TRACE_DIR, exist_ok=True)
    suffix = time.time_ns() % 1_000_000_000
    filename = (
        f"teleop_{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{suffix:09d}.jsonl"
    )
    return os.path.join(TRACE_DIR, filename)


class LatencyMonitor:
    """Collect timing samples and write JSONL without blocking control."""

    METRICS = (
        "leader_read_ms",
        "compute_ms",
        "servoj_rtt_ms",
        "whole_loop_ms",
        "actual_loop_hz",
    )

    def __init__(self, trace_path, session_metadata=None):
        self.trace_path = trace_path
        self.session_metadata = session_metadata or {}
        self._trace_file = open(
            trace_path,
            "w",
            encoding="utf-8",
        )
        self._trace_count = 0
        self._trace_error = None
        self._last_trace_flush = time.monotonic()
        self._samples = queue.Queue(maxsize=TRACE_QUEUE_SIZE)
        self._stop = threading.Event()
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._write_metadata()
        self._thread = threading.Thread(
            target=self._run,
            name="teleop-latency-monitor",
            daemon=True,
        )
        self._thread.start()

    def submit(self, sample):
        try:
            self._samples.put_nowait(sample)
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if not self._thread.is_alive():
            try:
                self._trace_file.flush()
                self._trace_file.close()
            except Exception:
                pass

    def _write_metadata(self):
        metadata = {
            "record_type": "metadata",
            "schema_version": TRACE_SCHEMA_VERSION,
            "wall_time_ns": time.time_ns(),
            "config": {
                "dt_s": DT,
                "max_offset_deg": list(MAX_OFFSET_DEG),
                "max_speed_deg_s": list(MAX_SPEED_DEG_S),
                "leader_stale_timeout_s": LEADER_STALE_TIMEOUT,
                "servo_t": SERVO_T,
                "servo_aheadtime": SERVO_AHEADTIME,
                "servo_gain": SERVO_GAIN,
                "map_sign": list(MAP_SIGN),
                "scale": list(SCALE),
            },
            "session": self.session_metadata,
        }
        self._trace_file.write(
            json.dumps(metadata, separators=(",", ":")) + "\n"
        )
        self._trace_file.flush()

    def _write_sample(self, sample):
        if self._trace_error is not None:
            return

        try:
            self._trace_file.write(
                json.dumps(
                    sample,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            self._trace_count += 1
            now = time.monotonic()
            if (
                self._trace_count % 32 == 0
                or now - self._last_trace_flush >= TRACE_FLUSH_PERIOD_S
            ):
                self._trace_file.flush()
                self._last_trace_flush = now
        except Exception as exc:
            self._trace_error = repr(exc)

    def _take_dropped(self):
        with self._drop_lock:
            dropped = self._dropped
            self._dropped = 0
        return dropped

    @staticmethod
    def _p95(values):
        values = sorted(values)
        return values[int(0.95 * (len(values) - 1))]

    def _print_summary(self, samples, dropped):
        if not samples and not dropped:
            return

        print()
        print(
            f"[latency] samples={len(samples)} "
            f"dropped={dropped} "
            f"trace={self._trace_count}"
        )

        if self._trace_error is not None:
            print(f"  trace_error: {self._trace_error}")

        for metric in self.METRICS:
            values = [
                sample[metric]
                for sample in samples
                if metric in sample
            ]
            if not values:
                continue

            print(
                f"  {metric}: "
                f"mean={statistics.fmean(values):.3f} "
                f"median={statistics.median(values):.3f} "
                f"p95={self._p95(values):.3f} "
                f"max={max(values):.3f}"
            )

    def _run(self):
        samples = []
        window_start = time.monotonic()

        while not self._stop.is_set() or not self._samples.empty():
            timeout = min(
                0.05,
                max(
                    0.001,
                    TIMING_SUMMARY_PERIOD_S
                    - (time.monotonic() - window_start),
                ),
            )

            try:
                sample = self._samples.get(timeout=timeout)
                self._write_sample(sample)
                samples.append(sample)
                self._samples.task_done()
            except queue.Empty:
                pass

            if time.monotonic() - window_start >= TIMING_SUMMARY_PERIOD_S:
                self._print_summary(samples, self._take_dropped())
                samples = []
                window_start = time.monotonic()

        self._print_summary(samples, self._take_dropped())


# ============================================================
# START
# ============================================================

print()
print("==========================================")
print(" X-Trainer -> Nova 6-DOF ServoJ")
print("==========================================")
print()
print("REAL ROBOT MOTION ENABLED")
print()
print("Limits:")
print("  offset : +/-20 deg per joint")
print("  speed  : 15 deg/s per joint")
print("  rate   : ~33 Hz")
print()


# ============================================================
# Leader setup
# ============================================================

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
        f"Cannot set Leader baud "
        f"{LEADER_BAUD}"
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
        result != COMM_SUCCESS
        or error != 0
    ):
        leader_port.closePort()

        raise RuntimeError(
            f"Torque OFF failed "
            f"for ID {motor_id}: "
            f"result={result}, "
            f"error={error}"
        )

print("Leader unlocked.")


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

    if not reader.addParam(
        motor_id
    ):
        raise RuntimeError(
            f"Cannot add Leader ID "
            f"{motor_id}"
        )


def get_leader_deg():

    result = reader.txRxPacket()

    if result != COMM_SUCCESS:
        return None

    joints = []

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

        q_rad = (
            raw_rad - OFFSETS[i]
        ) * SIGNS[i]

        joints.append(
            math.degrees(q_rad)
        )

    return joints


# ============================================================
# Nova connection
# ============================================================

nova = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

nova.settimeout(1.0)

servo_started = False
latency_monitor = None


try:

    print(
        f"Connecting Nova "
        f"{NOVA_IP}:{NOVA_PORT} ..."
    )

    nova.connect(
        (NOVA_IP, NOVA_PORT)
    )

    print("Nova connected.")


    # --------------------------------------------------------
    # Initial state check
    # --------------------------------------------------------

    mode_resp = send_dobot(
        nova,
        "RobotMode()",
    )

    print(
        "RobotMode:",
        mode_resp.strip(),
    )

    mode = parse_brace_numbers(
        mode_resp
    )

    if (
        len(mode) != 1
        or int(mode[0]) != 5
    ):
        raise RuntimeError(
            "Nova must start in "
            "RobotMode 5 (ENABLE idle). "
            "No ServoJ sent."
        )


    print(
        "GetErrorID:",
        send_dobot(
            nova,
            "GetErrorID()",
        ).strip(),
    )


    nova_zero = parse_brace_numbers(
        send_dobot(
            nova,
            "GetAngle()",
        )
    )

    if len(nova_zero) != 6:
        raise RuntimeError(
            f"Invalid Nova joints: "
            f"{nova_zero}"
        )


    # --------------------------------------------------------
    # Leader initial read
    # --------------------------------------------------------

    leader_zero = None

    deadline = (
        time.monotonic() + 2.0
    )

    while (
        leader_zero is None
        and time.monotonic() < deadline
    ):
        leader_zero = get_leader_deg()
        time.sleep(0.01)

    if leader_zero is None:
        raise RuntimeError(
            "Cannot read Leader."
        )


    print()
    print("Nova:")
    print(
        "  "
        + " ".join(
            f"{x:7.2f}"
            for x in nova_zero
        )
    )

    print("Leader:")
    print(
        "  "
        + " ".join(
            f"{x:7.2f}"
            for x in leader_zero
        )
    )


    # --------------------------------------------------------
    # User arm confirmation
    # --------------------------------------------------------

    print()
    print("------------------------------------------")
    print("6-DOF REAL SERVO TEST")
    print()
    print("保持 Leader 当前姿态。")
    print("确保 Nova 周围无人、无障碍。")
    print("手放在急停附近。")
    print()
    print("Enter = 建立当前主从 anchor 并开始")
    print("Ctrl+C = 停止")
    print("------------------------------------------")

    input()


    # --------------------------------------------------------
    # Re-anchor immediately before servo
    # --------------------------------------------------------

    leader_zero = get_leader_deg()

    if leader_zero is None:
        raise RuntimeError(
            "Leader unavailable before start."
        )

    nova_zero = parse_brace_numbers(
        send_dobot(
            nova,
            "GetAngle()",
        )
    )

    if len(nova_zero) != 6:
        raise RuntimeError(
            "Nova anchor read failed."
        )


    commanded = nova_zero.copy()

    last_loop = time.monotonic()
    last_leader_ok = (
        time.monotonic()
    )

    trace_path = make_trace_path()
    latency_monitor = LatencyMonitor(
        trace_path,
        session_metadata={
            "leader_port": LEADER_PORT,
            "nova_ip": NOVA_IP,
            "nova_port": NOVA_PORT,
            "leader_zero_deg": list(leader_zero),
            "nova_zero_deg": list(nova_zero),
        },
    )
    print(f"Telemetry trace: {trace_path}")
    next_display = time.monotonic()
    servo_started = True

    print()
    print("===== 6-DOF SERVO ACTIVE =====")
    print()


    # ========================================================
    # Main Servo Loop
    # ========================================================

    while True:

        cycle_start = (
            time.monotonic_ns()
        )

        leader_read_start = cycle_start
        leader_now = get_leader_deg()
        leader_read_done = time.monotonic_ns()


        # ----------------------------------------------------
        # Watchdog
        # ----------------------------------------------------

        if leader_now is None:

            if (
                time.monotonic()
                - last_leader_ok
                > LEADER_STALE_TIMEOUT
            ):

                print()
                print(
                    "Leader stale -> STOP"
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

                cycle_end = time.monotonic_ns()
                latency_monitor.submit(
                    {
                        "record_type": "sample",
                        "wall_time_ns": time.time_ns(),
                        "t_cycle_start": cycle_start,
                        "t_leader_read_start": leader_read_start,
                        "t_leader_read_done": leader_read_done,
                        "t_target_compute_done": None,
                        "t_servoj_send": None,
                        "t_servoj_response": None,
                        "t_cycle_end": cycle_end,
                        "leader_read_ms": (
                            leader_read_done - leader_read_start
                        ) / 1_000_000,
                        "whole_loop_ms": (
                            cycle_end - cycle_start
                        ) / 1_000_000,
                        "actual_loop_hz": (
                            1_000_000_000
                            / max(cycle_end - cycle_start, 1)
                        ),
                        "status": "leader_stale",
                    }
                )
                break

            time.sleep(0.005)
            cycle_end = time.monotonic_ns()
            latency_monitor.submit(
                {
                    "record_type": "sample",
                    "wall_time_ns": time.time_ns(),
                    "t_cycle_start": cycle_start,
                    "t_leader_read_start": leader_read_start,
                    "t_leader_read_done": leader_read_done,
                    "t_target_compute_done": None,
                    "t_servoj_send": None,
                    "t_servoj_response": None,
                    "t_cycle_end": cycle_end,
                    "leader_read_ms": (
                        leader_read_done - leader_read_start
                    ) / 1_000_000,
                    "whole_loop_ms": (
                        cycle_end - cycle_start
                    ) / 1_000_000,
                    "actual_loop_hz": (
                        1_000_000_000
                        / max(cycle_end - cycle_start, 1)
                    ),
                    "status": "leader_unavailable",
                }
            )
            continue


        last_leader_ok = (
            time.monotonic()
        )


        # ----------------------------------------------------
        # Actual loop dt
        # ----------------------------------------------------

        now = time.monotonic()

        actual_dt = max(
            now - last_loop,
            0.001,
        )

        last_loop = now


        # ----------------------------------------------------
        # Calculate all 6 desired joints
        # ----------------------------------------------------

        desired = [0.0] * 6
        delta = [0.0] * 6

        for i in range(6):

            d = (
                leader_now[i]
                - leader_zero[i]
            )

            d *= (
                MAP_SIGN[i]
                * SCALE[i]
            )

            d = clamp(
                d,
                -MAX_OFFSET_DEG[i],
                MAX_OFFSET_DEG[i],
            )

            delta[i] = d

            desired[i] = (
                nova_zero[i] + d
            )


        # ----------------------------------------------------
        # Per-joint rate limiter
        # ----------------------------------------------------

        for i in range(6):

            max_step = (
                MAX_SPEED_DEG_S[i]
                * actual_dt
            )

            error = (
                desired[i]
                - commanded[i]
            )

            step = clamp(
                error,
                -max_step,
                max_step,
            )

            commanded[i] += step

        target_compute_done = time.monotonic_ns()


        # ----------------------------------------------------
        # ServoJ
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

        servo_timing = {}
        response = send_dobot(
            nova,
            cmd,
            timing=servo_timing,
        )

        error_id = parse_error_id(
            response
        )

        if error_id != 0:

            print()
            print(
                "ServoJ rejected:"
            )
            print(response)

            cycle_end = time.monotonic_ns()
            latency_monitor.submit(
                {
                    "record_type": "sample",
                    "wall_time_ns": time.time_ns(),
                    "t_cycle_start": cycle_start,
                    "t_leader_read_start": leader_read_start,
                    "t_leader_read_done": leader_read_done,
                    "t_target_compute_done": target_compute_done,
                    "t_servoj_send": servo_timing["t_servoj_send"],
                    "t_servoj_response": servo_timing["t_servoj_response"],
                    "t_cycle_end": cycle_end,
                    "leader_deg": list(leader_now),
                    "delta_deg": list(delta),
                    "desired_deg": list(desired),
                    "commanded_deg": list(commanded),
                    "servoj_command": cmd,
                    "leader_read_ms": (
                        leader_read_done - leader_read_start
                    ) / 1_000_000,
                    "compute_ms": (
                        target_compute_done - leader_read_done
                    ) / 1_000_000,
                    "servoj_rtt_ms": (
                        servo_timing["t_servoj_response"]
                        - servo_timing["t_servoj_send"]
                    ) / 1_000_000,
                    "whole_loop_ms": (
                        cycle_end - cycle_start
                    ) / 1_000_000,
                    "actual_loop_hz": (
                        1_000_000_000
                        / max(cycle_end - cycle_start, 1)
                    ),
                    "status": "servoj_rejected",
                }
            )

            break


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        if time.monotonic() >= next_display:
            print(
                "\rΔLeader: "
                + " ".join(
                    f"{x:+5.1f}"
                    for x in delta
                )
                + " | Cmd: "
                + " ".join(
                    f"{x:7.1f}"
                    for x in commanded
                ),
                end="",
                flush=True,
            )
            next_display = time.monotonic() + DISPLAY_PERIOD_S


        # ----------------------------------------------------
        # ~33 Hz
        # ----------------------------------------------------

        elapsed = (
            time.monotonic_ns()
            - cycle_start
        ) / 1_000_000_000

        remaining = (
            DT - elapsed
        )

        if remaining > 0:
            time.sleep(
                remaining
            )

        cycle_end = time.monotonic_ns()
        latency_monitor.submit(
            {
                "record_type": "sample",
                "wall_time_ns": time.time_ns(),
                "t_cycle_start": cycle_start,
                "t_leader_read_start": leader_read_start,
                "t_leader_read_done": leader_read_done,
                "t_target_compute_done": target_compute_done,
                "t_servoj_send": servo_timing["t_servoj_send"],
                "t_servoj_response": servo_timing["t_servoj_response"],
                "t_cycle_end": cycle_end,
                "leader_deg": list(leader_now),
                "delta_deg": list(delta),
                "desired_deg": list(desired),
                "commanded_deg": list(commanded),
                "actual_dt_s": actual_dt,
                "servoj_command": cmd,
                "leader_read_ms": (
                    leader_read_done - leader_read_start
                ) / 1_000_000,
                "compute_ms": (
                    target_compute_done - leader_read_done
                ) / 1_000_000,
                "servoj_rtt_ms": (
                    servo_timing["t_servoj_response"]
                    - servo_timing["t_servoj_send"]
                ) / 1_000_000,
                "whole_loop_ms": (
                    cycle_end - cycle_start
                ) / 1_000_000,
                "actual_loop_hz": (
                    1_000_000_000
                    / max(cycle_end - cycle_start, 1)
                ),
                "status": "ok",
            }
        )


except KeyboardInterrupt:

    print()
    print("Ctrl+C received.")


except Exception as e:

    print()
    print("ERROR:", e)


finally:

    if servo_started:

        try:

            print()
            print("Sending Stop() ...")

            print(
                send_dobot(
                    nova,
                    "Stop()",
                ).strip()
            )

        except Exception as e:

            print(
                "Stop failed:",
                e,
            )

    if latency_monitor is not None:
        latency_monitor.stop()


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
