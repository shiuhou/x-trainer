# Personal publication boundary

The personal fork is a sanitized research checkout. It does not include local
Dobot settings, camera serials, service credentials, TLS private keys or raw
teleoperation traces.

Before running a hardware-facing tool, copy
`source/leisaac/leisaac/xtrainer_utils/utils/dobot_config/dobot_settings.example.ini`
to `dobot_settings.ini` and fill in the local values. Keep that file untracked.
Live scripts use explicit placeholder endpoint values in the published checkout;
configure the actual Nova address and serial devices locally.

The upstream X-LeVR key and certificate were removed from the entire published
history. Any deployment that used those files must revoke and replace them
outside this repository.
