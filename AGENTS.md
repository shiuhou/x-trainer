# Agent Instructions

This is the sanitized public X-Trainer research checkout. It contains Isaac
Lab/LeIsaac simulation, Leader teleoperation, Nova pilot tooling, gripper
helpers and episode QA utilities. Read `README.md`, `PUBLICATION.md`, and the
specific `README_NOVA_TELEOP.md` or `docs/` checkpoint before changing behavior.

## Publication and hardware boundaries

- Keep local settings, passcodes, service credentials, TLS keys/certificates,
  camera serials, raw traces and machine-specific endpoints out of commits.
- Hardware-facing scripts are opt-in research tools, not a general safety API.
  Do not connect to, enable, power, clear errors on, or move Nova or a Leader
  unless the operator has current authorization for the exact procedure.
- Do not add automatic `PowerOn`, `EnableRobot`, `ClearError`, `RunTo`,
  `MovL`, `ServoP` or `ServoJ` behavior. Preserve explicit confirmations and
  stop/failure handling.
- Treat published endpoint and serial placeholders as required local
  configuration. Copy `dobot_settings.example.ini` to an untracked local
  settings file before hardware use.
- Do not treat a teleoperation pilot, dry run, fake device or simulator result
  as a hardware safety certification.
- For Dobot TCP/IP facts, consult the workspace reference
  `/home/dsa/project/dobot/dobot_tcp_ip_reference.md`; use the original PDF
  when the Markdown extraction is ambiguous.

## Validation

- Keep offline parser, controller and episode QA tests network-independent.
- Compile and run focused tests for changed modules before publication.
- Preserve raw episode evidence and its hashes; write derived output to a new
  directory rather than overwriting a frozen checkpoint.
- Scan staged changes for secrets and local hardware identifiers before push.

The workspace-level coordination rules are in the parent repository's
`AGENTS.md`, `README.md`, `HANDOFF.md` and `components.lock.yaml`.
