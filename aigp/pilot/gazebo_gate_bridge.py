#!/usr/bin/env python3
"""
Stream Gazebo/static gate centers as competition-style MAVLink track data.

This matches the parser in mavlink_rx.py:
  DATA_TRANSMISSION_HANDSHAKE:
    width   -> track-data transfer_id
    packets -> number of ENCAPSULATED_DATA chunks

  ENCAPSULATED_DATA.data:
    byte 0      -> data_type, 2 means track info
    bytes 1..2 -> uint16 transfer_id
    bytes 3..  -> chunk of the track payload

Track payload:
  uint16 num_gates
  repeated gate records: <Hfffffffff
    gate_id, position_ned_x/y/z, orientation_ned_w/x/y/z, width, height
"""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib


ENCAPSULATED_TRACK_INFO_MSG_ID = 2
ENCAPSULATED_DATA_BYTES = 253
TRACK_CHUNK_HEADER_BYTES = 3
TRACK_CHUNK_PAYLOAD_BYTES = ENCAPSULATED_DATA_BYTES - TRACK_CHUNK_HEADER_BYTES


@dataclass(frozen=True)
class Gate:
    gate_id: int
    position_ned: tuple[float, float, float]
    orientation_ned_wxyz: tuple[float, float, float, float]
    width: float
    height: float


def position_to_ned(
    position: Iterable[float],
    frame: str,
) -> tuple[float, float, float]:
    x, y, z = [float(v) for v in position]

    if frame == "ned":
        return x, y, z
    if frame == "neu":
        return x, y, -z
    if frame == "enu":
        return y, x, -z

    raise ValueError(f"Unsupported input frame: {frame}")


def parse_gate_arg(text: str, input_frame: str) -> Gate:
    """
    Parse one CLI gate.

    Format:
      id,x,y,z,width,height
      id,x,y,z,width,height,qw,qx,qy,qz

    The x/y/z values are interpreted using --input-frame.
    """
    parts = [item.strip() for item in text.split(",")]

    if len(parts) not in (6, 10):
        raise argparse.ArgumentTypeError(
            "--gate must be id,x,y,z,width,height"
            " or id,x,y,z,width,height,qw,qx,qy,qz"
        )

    try:
        gate_id = int(parts[0])
        position = position_to_ned((parts[1], parts[2], parts[3]), input_frame)
        width = float(parts[4])
        height = float(parts[5])

        if len(parts) == 10:
            orientation = tuple(float(v) for v in parts[6:10])
        else:
            orientation = (1.0, 0.0, 0.0, 0.0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    return Gate(
        gate_id=gate_id,
        position_ned=position,
        orientation_ned_wxyz=orientation,
        width=width,
        height=height,
    )


def load_gates_toml(path: Path, default_input_frame: str) -> list[Gate]:
    """
    Load gates from TOML.

    Supported shape:

      [[gate]]
      id = 0
      position_ned = [5.0, 0.0, -1.5]
      orientation_ned_wxyz = [1.0, 0.0, 0.0, 0.0]
      width = 1.5
      height = 1.5

    position_neu, position_enu, or generic position plus frame are also accepted.
    """
    with path.open("rb") as f:
        data = tomllib.load(f)

    gate_entries = data.get("gate", data.get("gates"))

    if not isinstance(gate_entries, list):
        raise ValueError(
            f"{path} must contain [[gate]] entries or a gates array"
        )

    gates: list[Gate] = []

    for index, entry in enumerate(gate_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Gate entry {index} is not a table")

        gate_id = int(entry.get("id", entry.get("gate_id", index)))

        if "position_ned" in entry:
            position = position_to_ned(entry["position_ned"], "ned")
        elif "position_neu" in entry:
            position = position_to_ned(entry["position_neu"], "neu")
        elif "position_enu" in entry:
            position = position_to_ned(entry["position_enu"], "enu")
        elif "position" in entry:
            frame = str(entry.get("frame", default_input_frame)).lower()
            position = position_to_ned(entry["position"], frame)
        else:
            raise ValueError(f"Gate {gate_id} is missing a position")

        orientation = tuple(
            float(v)
            for v in entry.get("orientation_ned_wxyz", [1.0, 0.0, 0.0, 0.0])
        )

        if len(orientation) != 4:
            raise ValueError(f"Gate {gate_id} orientation must have 4 values")

        gates.append(
            Gate(
                gate_id=gate_id,
                position_ned=position,
                orientation_ned_wxyz=orientation,
                width=float(entry.get("width", 1.5)),
                height=float(entry.get("height", 1.5)),
            )
        )

    return gates


def pack_track_payload(gates: list[Gate]) -> bytes:
    payload = bytearray()
    payload.extend(struct.pack("<H", len(gates)))

    for gate in gates:
        payload.extend(
            struct.pack(
                "<Hfffffffff",
                gate.gate_id,
                gate.position_ned[0],
                gate.position_ned[1],
                gate.position_ned[2],
                gate.orientation_ned_wxyz[0],
                gate.orientation_ned_wxyz[1],
                gate.orientation_ned_wxyz[2],
                gate.orientation_ned_wxyz[3],
                gate.width,
                gate.height,
            )
        )

    return bytes(payload)


def build_encapsulated_packets(payload: bytes, transfer_id: int) -> list[bytes]:
    packets: list[bytes] = []

    for offset in range(0, len(payload), TRACK_CHUNK_PAYLOAD_BYTES):
        chunk = payload[offset:offset + TRACK_CHUNK_PAYLOAD_BYTES]
        packet = struct.pack(
            "<BH",
            ENCAPSULATED_TRACK_INFO_MSG_ID,
            transfer_id,
        ) + chunk

        if len(packet) > ENCAPSULATED_DATA_BYTES:
            raise RuntimeError("Track packet exceeded ENCAPSULATED_DATA size")

        packet += bytes(ENCAPSULATED_DATA_BYTES - len(packet))
        packets.append(packet)

    return packets


def send_track(conn, gates: list[Gate], transfer_id: int) -> None:
    payload = pack_track_payload(gates)
    packets = build_encapsulated_packets(payload, transfer_id)

    # The receiver only uses width as transfer_id and packets as the count.
    conn.mav.data_transmission_handshake_send(
        0,                          # type, unused by mavlink_rx.py
        len(payload),               # size
        transfer_id,                # width repurposed as transfer_id
        len(gates),                 # height, informational only here
        len(packets),               # packets
        TRACK_CHUNK_PAYLOAD_BYTES,  # payload bytes per packet
        0,                          # jpg_quality, unused
    )

    for seqnr, packet in enumerate(packets):
        conn.mav.encapsulated_data_send(seqnr, list(packet))


def print_gate_summary(gates: list[Gate]) -> None:
    print(f"Loaded {len(gates)} gate(s):", flush=True)
    for gate in gates:
        x, y, z = gate.position_ned
        print(
            f"  gate {gate.gate_id}: "
            f"ned=({x:.3f}, {y:.3f}, {z:.3f}) "
            f"size=({gate.width:.3f}, {gate.height:.3f})",
            flush=True,
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send competition-format track_gates over MAVLink UDP for "
            "pilot."
        )
    )
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14540)
    parser.add_argument("--hz", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--transfer-id", type=int, default=1)
    parser.add_argument(
        "--input-frame",
        choices=("ned", "neu", "enu"),
        default="ned",
        help=(
            "Frame for --gate x/y/z values. ned is MAVLink local NED; "
            "neu is z-up with north/east axes; enu is common Gazebo world."
        ),
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        help=(
            "Repeatable gate: id,x,y,z,width,height[,qw,qx,qy,qz]. "
            "x/y/z use --input-frame."
        ),
    )
    parser.add_argument(
        "--gates-config",
        type=Path,
        help="TOML file containing [[gate]] entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pack and print the payload summary without importing pymavlink.",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    gates: list[Gate] = []

    if args.gates_toml is not None:
        gates.extend(load_gates_toml(args.gates_toml, args.input_frame))

    for gate_text in args.gate:
        gates.append(parse_gate_arg(gate_text, args.input_frame))

    if not gates:
        parser.error("Provide at least one --gate or --gates-config file")

    if args.transfer_id < 0 or args.transfer_id > 65535:
        parser.error("--transfer-id must fit in uint16")

    if args.hz <= 0.0:
        parser.error("--hz must be positive")

    print_gate_summary(gates)

    payload = pack_track_payload(gates)
    packets = build_encapsulated_packets(payload, args.transfer_id)
    print(
        "Track payload:",
        f"bytes={len(payload)}",
        f"packets={len(packets)}",
        f"transfer_id={args.transfer_id}",
        flush=True,
    )

    if args.dry_run:
        return 0

    from pymavlink import mavutil

    conn = mavutil.mavlink_connection(
        f"udpout:{args.ip}:{args.port}",
        source_system=250,
        source_component=190,
    )

    print(
        f"Streaming track data to {args.ip}:{args.port} at {args.hz:.3f} Hz",
        flush=True,
    )

    period_s = 1.0 / args.hz
    send_count = 0

    try:
        while True:
            send_track(conn, gates, args.transfer_id)
            send_count += 1
            print(f"sent track transfer #{send_count}", flush=True)

            if args.once:
                break

            time.sleep(period_s)
    except KeyboardInterrupt:
        print("Stopping gate bridge.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
