# Competition Adapter Phase 8.5C UDP Vision Loopback

Status: implemented for generated mock-image loopback only.

Date: 2026-06-12

## Scope

Phase 8.5C validates the competition vision packet receiver boundary without
starting PX4, Gazebo, the competition simulator, `CompetitionRunner`,
`AutonomyAPI`, perception, telemetry, or command paths.

The implemented tool is:

- `autonomy_core/tools/competition_vision_udp_loopback.py`

It exercises:

```text
generated mock image
  -> JPEG encode
  -> VADR <IHHIIQ> packetization
  -> UDP send to 127.0.0.1:5600
  -> UDP receive on 0.0.0.0:5600
  -> CompetitionImageAdapter.process_packet(...)
  -> CompetitionCameraFrame
```

## What It Proves

- The receiver can bind `0.0.0.0:5600` when explicitly run.
- Mock VADR UDP datagrams reach the local receiver as real UDP packets.
- The exact `<IHHIIQ` vision header is parsed by the existing image adapter.
- JPEG chunks are reassembled by `frame_id`.
- Completed JPEG frames are decoded by `CompetitionImageAdapter`.
- The completed frame uses official camera metadata:
  `640x360`, `fx=fy=320`, `cx=320`, `cy=180`, and zero distortion.
- `gazebo_pose` and `image_pose_snapshot` remain `None`.

## Manual Smoke Command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.tools.competition_vision_udp_loopback \
  --bind-host 0.0.0.0 \
  --send-host 127.0.0.1 \
  --port 5600 \
  --frames 1 \
  --timeout-s 5 \
  --save-mock-image /tmp/competition_vision_loopback_mock.jpg \
  --save-decoded-image /tmp/competition_vision_loopback_decoded.jpg
```

Expected key fields:

```text
packets_sent > 0
packets_received == packets_sent
frames_completed == 1
image_shape == [360, 640, 3]
gazebo_pose == null
image_pose_snapshot == null
mock_image_path == /tmp/competition_vision_loopback_mock.jpg
decoded_image_path == /tmp/competition_vision_loopback_decoded.jpg
```

## Tests

`tests/test_competition_vision_udp_loopback.py` covers deterministic,
no-socket behavior:

- importing the module does not load `cv2`, `pymavlink`, `mavsdk`, or `rclpy`;
- generated mock image shape, dtype, and nonuniform pattern;
- VADR packet header fields and chunk counts;
- direct packet processing through `CompetitionImageAdapter`;
- malformed packet rejection accounting;
- CLI defaults for `0.0.0.0:5600` receive and `127.0.0.1:5600` send.

Unit tests intentionally do not bind `5600`, because that port may be in use.
The real UDP socket path is covered by the explicit manual smoke command.

## Non-Claims

- This does not start or require PX4, Gazebo, MAVSDK, ROS, or the competition
  simulator.
- This does not read Gazebo camera frames.
- This does not add live competition vision transport to `CompetitionRunner`.
- This does not call `CompetitionRunner`, `AutonomyAPI`, YOLO, PnP, planner,
  controller, state adapter, command adapter, or any command publication path.
- This does not satisfy Phase 4B or Phase 9.
- This does not claim telemetry readiness, command readiness, race readiness,
  or competition readiness.
