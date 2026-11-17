# NeuralStick Bridge macOS arm64 toolchain

## Design

```text
macOS arm64 process
        |
        +-- read_ncs2_usb.py + Homebrew libusb
        |
        +-- OpenVINO 2022.3.2 arm64 runtime
                |
                +-- MYRIAD plugin / XLink / MVNC
                        |
                        +-- NCS2 USB firmware and MyriadX
```

Intel removed MYRIAD support from OpenVINO after the 2022.3 line. This project therefore builds a pinned compatibility runtime while keeping the host-facing layer modern and small: native process checks, explicit paths, a versioned worker protocol, Python and Swift clients, and an arm64 architecture audit.

Installing the current OpenVINO package does not make an NCS2 discoverable. The wrapper in this repository is what carries the old device support into a current macOS application.

## Pinned inputs

- OpenVINO source: `releases/2022/3`, commit `6fbc44f307068367bef029f760e26058147b6d1b`.
- Target architecture: `arm64`.
- Required build options: `ENABLE_INTEL_MYRIAD=ON`, `ENABLE_INTEL_MYRIAD_COMMON=ON`.
- Firmware archives: `firmware_usb-ma2x8x_20230418_40.zip` and `firmware_pcie-ma2x8x_20230418_40.zip`, obtained by `./ncs2 fetch` and kept locally.

Exact URLs, archive hashes, extracted-file hashes, and submodule revisions are in `toolchain.lock.json`. The bootstrap script verifies existing source and downloaded bytes before using them. Firmware and vendor source stay under ignored local directories.

`scripts/build_runtime.sh` contains the tested CMake configuration. `THREADING=SEQ` makes the earlier fallback from missing TBB explicit. `CMAKE_POLICY_VERSION_MINIMUM=3.5` lets the older project configure with current CMake. `ENABLE_PKGCONFIG_GEN=OFF` avoids pkg-config failures on paths containing spaces.

The MYRIAD plugin is named `libopenvino_intel_myriad_plugin.so` even on macOS; it is a Mach-O arm64 library. `./ncs2 check` recursively audits non-system dependencies and rejects universal or x86_64 slices. `./ncs2 check --json` emits one JSON document with architecture, dependency edges, and SHA-256 values.

The probe uses an explicit `plugins.xml` path. The launcher resolves paths from its own location, so it can be invoked outside the repository directory. Moving an already-built checkout can invalidate CMake caches and rpaths; rebuild after relocation.

## Native host boundary

The old runtime remains inside dedicated worker executables. Applications communicate through a bounded JSON-lines protocol instead of linking OpenVINO or sharing its C++ ABI. This lets a current Python environment or a native Swift application use the device without an x86_64 dependency.

This removes the x86_64 host dependency for the demonstrated inference path. It does not replace Intel firmware, reimplement the device compiler, or establish arbitrary operator compatibility. The pinned Xbyak source is a build input, not a loaded x86_64 library.

## Diagnostics

`./ncs2 doctor` performs a read-only preflight without opening USB. It checks the native host, command-line tools, the pinned OpenVINO checkout, runtime libraries, firmware output hashes, expected build products (including the offline model inspector), and the shared device-lock path. Runtime and libusb checks verify pure arm64 slices. `--json` emits one JSON report on stdout; log notices go to stderr. `./ncs2 doctor --hardware` additionally opens USB and asks MYRIAD for devices, without inference. Exit 0 means the selected checks passed; exit 2 means one or more required checks need attention.

`./ncs2 capabilities --json` emits the supported model and worker contract without hardware access. `./ncs2 inspect --model MODEL.xml` reads model metadata offline and reports schema compatibility. The inspector links the pinned OpenVINO model reader but does not load the MYRIAD plugin, acquire the device lock or access USB. A compatible report checks structural limits; MYRIAD graph compilation still occurs at worker startup.

The probe, schema-1 worker, schema-2 worker, and async SDK use the same nonblocking advisory lock. Set `NCS2_DEVICE_LOCK` to select an alternate lock file when coordinating multiple processes. A second NeuralStick Bridge process reports `device_busy` and does not start a MYRIAD session. `--validate-only` worker checks do not acquire the device lock.

- Exit 0 means the requested checks passed. Exit 2 means no MYRIAD device, missing runtime or probe, or no usable matching USB device. Other nonzero codes report build or validation failures.
- `NCS2_LOG_LEVEL=LOG_DEBUG ./ncs2 smoke` sets the plugin's public log level. This revision supports `LOG_NONE`, `LOG_ERROR`, `LOG_WARNING`, `LOG_INFO`, `LOG_DEBUG`, and `LOG_TRACE`.
- `./ncs2 usb --show-serial` is an explicit local-only diagnostic option; serial numbers are hidden by default.
- `NCS2_LIBUSB` may specify an absolute libusb path. `NCS2_JOBS` controls build parallelism; the default is 4.
- The logger executes argv without shell evaluation, buffers output before appending, uses a nonblocking lock, and retains at most 2 MiB per command.

USB access can differ between a sandboxed process and a normal macOS terminal. Run hardware probes from a normal terminal when libusb reports no device while `ioreg` shows the NCS2.

## Scope

The public toolchain covers dependency bootstrap, native builds, USB descriptor reading, MYRIAD discovery, offline IR inspection, the capability matrix, a deterministic ReLU smoke test and benchmark, schema-1 and schema-2 workers, Python/Swift clients, and architecture checks. Reproduction suites, soak tests, standalone performance laboratories, lifecycle tracing, and application experiments remain outside the default toolchain. Candidate workloads and measurement definitions are documented in `docs/PERFORMANCE_WORKLOADS.md`.
