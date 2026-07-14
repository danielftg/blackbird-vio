# Hardware — platform component map

A general map of the platform: each component's internal structure and how it connects to the others.
The **terms, the components, where to find more** — not a deep-dive. The *chosen* parts (specific SoC, sync plan)
are a design decision and live in [[design/hardware]]; this file is the general model the design is chosen from.

---

## The components and how they interconnect

Three networks tie the platform together: a **power bus** (energy), a **data/sensor path** (bits in), and a **comms/control path** (commands and telemetry). Each component below sits on one or more.

1. **Battery** — pack of cells (series→voltage, parallel→capacity), connector, straps. The energy source; top of the power bus.
2. **Stereo camera rig** — two synchronized image sensors at a fixed baseline. The perception sensor; feeds the data path.
3. **Motors + propellers** — brushless motors turning props; the actuators. Draw the most power, off the power bus.
4. **ESC** (electronic speed controller) — drives each motor; its **bus capacitor** absorbs the switching current spikes. Sits between battery and motor; takes commands from the autopilot.
5. **Compute + autopilot** — the brain. Two roles: a **SoC** (vision, estimation, planning — see focus below) and a
   **flight-control MCU + IMU** (fast inner loop, motion sensing). On an integrated module these are one part; on a cheap build, two boards. The split is a design choice ([[design/hardware]]).
6. **Datalink** — radio up and down, plus antennas. Carries imagery to the operator (to draw the one-time mask), the command
   channel (priority switch, return), and telemetry.
7. **Power conditioning** — bus capacitor + rail filtering between battery and loads; gives each load a clean, steady voltage.
8. **Thermal management** — heatsink, fan, thermal interface. Without it onboard compute overheats and **throttles** (slows itself to survive).
9. **Vibration isolation** — damped mounts for the IMU / flight controller; without them the IMU reads propeller vibration as motion.
10. **Frame + casing** — plates, arms, walls, standoffs. The mechanical skeleton.
11. **Wiring + fastening** — cables, connectors, screws. Carries every power and data link above.

The rest of this file zooms into component **5**, the compute internals — the part with the most structure and the least obvious map.

---

## Compute internals

A computer is **memory** (where data sits) + **compute units** (what acts on it) + the **interconnect** between them.
Performance is mostly about keeping the compute units fed from memory.

### Memory hierarchy

Memory trades **speed against size**: fast memory is small and close to the compute unit; large memory is slow and far.
From fastest/smallest to slowest/largest:

- **Registers** — a handful of slots inside the compute unit; the values it's working on *right now*.
- **Cache** — small fast memory on the chip, in levels **L1 → L2 → L3** (each bigger and slower). Holds recently-used data
  so the unit doesn't wait on RAM. Cache misses are a dominant cost; laying data out to stay in cache is much of real performance.
- **RAM** (main memory) — the working set of the running program. Volatile (lost on power-off).
- **Disk / flash** — persistent storage (survives power-off). Slowest. On an embedded board this is flash, not a spinning disk.

→ The reference: [[drepper-memory]] (*What Every Programmer Should Know About Memory*) and [[hennessy-computers]] Ch. 2.

### Compute units — the specialization gradient

The same axis as [[design/hardware]]: **flexibility ↔ efficiency**. A unit that runs anything wastes energy; a unit built for
one job is efficient but rigid. `CPU → GPU → FPGA → TPU/NPU → ASIC`, more specialized (and efficient per watt) to the right.

- **CPU** (central processing unit) — runs *any* program, a few tasks at a time, fast on each. The general-purpose default. Internals below.
- **GPU** (graphics processing unit) — thousands of small units doing the *same* operation on *many* data items at once (parallel math).
  Good for per-pixel vision (LK, NCC) and the linear algebra in the EKF/solver.
- **FPGA** (field-programmable gate array) — reconfigurable hardware: you wire up a circuit for one task, near-ASIC efficient, and can
  rewire it later. (Full reconfig: a few ms, chip down; partial: faster, rest keeps running.) Good for fixed hot primitives (FAST, LK, NCC).
- **TPU / NPU** (tensor / neural processing unit) — a unit built specifically for neural-network math. The most specialized programmable point.
- **ASIC** — a circuit fixed in silicon for one job. Most efficient, zero flexibility.

→ [[hennessy-computers]]: the canon, by the creators of RISC. Ch. 7 (Domain-Specific Architectures) is the GPU/FPGA/TPU end, with the
Google TPU as the worked case. The flexibility↔efficiency reasoning and our SoC choice are in [[design/hardware]].

### Inside a CPU

- **Core** — one full compute unit. A **multi-core** chip has several, running genuinely in parallel.
- **Thread** — one independent stream of instructions. A core runs one (or, with hardware multithreading, two) at a time;
  the OS time-slices many software threads onto the cores.
- **Pipeline / instruction-level parallelism** — a core overlaps the stages of consecutive instructions (fetch, decode, execute)
  so several are in flight at once — a deeper source of speed than core count.
- **ISA** (instruction set architecture) — the vocabulary of operations the CPU understands; the contract between software and hardware.
  **RISC-V** is the open, royalty-free ISA (the example ISA in [[hennessy-computers]]; spec at riscv.org).

→ Digital-design level (gates → adder → core): [[harris-computers]]. Architecture level: [[hennessy-computers]] Ch. 3 (parallelism).

### The SoC

A **system-on-chip** puts CPU + GPU + (NPU, ISP, FPGA fabric) on one die sharing one RAM. Each job goes to the unit that fits it,
and **shared memory** means no copying data between units. This is the right architecture for our latency budget — argued in [[design/hardware]].

---

## Interfacing the machine — the OS

**Does a C++ program need an OS?** No. A **compiler** turns C++ source into raw machine instructions for the target ISA; those instructions run on the chip whether or not an OS is present — the language doesn't require one. What an OS *adds* is **services**: files, networking, threads, and on-demand memory, reached through the standard library. Bare-metal C++ works fine, but anything that asks the OS for a service (open a file, spawn a thread) has nothing to call, so you either avoid it or supply your own minimal runtime. So the same language spans both options below — rich on Linux, stripped on the MCU. 
→ [[bjarne-c++]] (the language; its concurrency primitives — `std::thread`, atomics — assume an OS underneath).

An **operating system** sits between hardware and program, managing memory, scheduling work onto cores, and talking to devices. Two ways to run our compute:

- **Linux** — a full OS. Splits into **kernel** (privileged core: drivers, scheduler, memory manager) and **user space** (your programs).
  The kernel owns the hardware; programs ask it for service. References: [[kerrisk-linux]] (programming against Linux),
  [[love-linux-kernel]] (how the kernel itself works).
  - **cgroupv2** (control groups, v2) — the kernel feature that **partitions resources** (CPU time, memory) between groups of processes,
    so vision can't starve flight control. The lever for our compute-budget allocation. (Kernel docs: `Documentation/admin-guide/cgroup-v2.rst`.)
- **Flashed / bare-metal** — no OS (or a tiny real-time one): the program runs directly on the chip. This is how the flight-control **MCU**   runs — small, deterministic, no Linux. Determinism and timing guarantees: [[buttazzo-real-time]].

The split mirrors component 5: **SoC on Linux** (rich, flexible, vision + planning) + **MCU flashed bare-metal** (tiny, hard-real-time, motor loop).

### Moving data: I/O, streaming, IPC

Three OS services that matter for us, all on the data path. The reference for all three on Linux is [[kerrisk-linux]]
(*The Linux Programming Interface*); the kernel mechanics behind them are in [[love-linux-kernel]].

- **I/O** (input/output) — how a program moves data to or from the outside: a device, a file, the network. The program issues a
  *read* / *write*; the kernel mediates and hands the bytes across. Every camera frame and every motor command is an I/O.
  → [[kerrisk-linux]] (file I/O, devices); electrical level of the bus underneath → [[horowitz-electronics]].
- **Streaming** — data that arrives as a *continuous flow* rather than one fixed lump (the stereo camera produces frames forever).
  Handled with **buffers / queues**: producer writes, consumer reads, with care that a slow consumer doesn't drop or stall the
  flow (**backpressure**). Our per-frame pipeline is a stream. → [[kerrisk-linux]] (pipes, buffering); timing under a deadline → [[buttazzo-real-time]].
- **IPC** (inter-process communication) — how two separate **processes** exchange data. (A process is one running program with its own
  isolated memory; a thread is a stream *within* one process — IPC crosses the process boundary, threads don't.) Forms: **pipes**   (a byte stream), **shared memory** (a region both map — fastest, zero-copy), **sockets** (also across machines), **message queues**.
  Relevant if vision and control run as separate processes, or compute talks to the autopilot.
  → [[kerrisk-linux]] (the IPC chapters); lock-free high-throughput hand-off → [[thompson-disruptor]] (the LMAX queue).

## Interfacing the rest of the platform from the computer

The compute board talks to the other components over two kinds of link:

- **Buses** — wires carrying bits between chips: low-rate sensor/control lines and high-rate streams (camera, motor telemetry).
  Electrical level: [[horowitz-electronics]]. The specific high-rate paths (camera stream, ESC telemetry) are a board-bring-up
  gap noted in [[design/hardware]], to fetch at part-selection time.
- **MAVLink** — the standard **message protocol** between an onboard computer and a flight controller/autopilot: it carries commands
  (setpoints, mode) down and telemetry (attitude, battery) up. The link from our SoC to the autopilot MCU. (Spec: mavlink.io.)

---

## Where to find more

| Topic | Source |
|---|---|
| Architecture, compute units, parallelism, DSA/TPU | [[hennessy-computers]] (H&P 6th ed., RISC-V) |
| Memory hierarchy & cache | [[drepper-memory]], [[hennessy-computers]] Ch. 2 |
| CPU from gates up | [[harris-computers]] |
| Linux programming / kernel | [[kerrisk-linux]], [[love-linux-kernel]] |
| I/O, streaming, IPC | [[kerrisk-linux]]; lock-free hand-off → [[thompson-disruptor]] |
| The language (C++) | [[bjarne-c++]] |
| cgroupv2 resource control | kernel `cgroup-v2.rst` |
| Real-time / bare-metal timing | [[buttazzo-real-time]] |
| Buses, electrical | [[horowitz-electronics]] |
| MAVLink protocol | mavlink.io |
| RISC-V ISA | riscv.org |
| **Our chosen hardware & rationale** | [[design/hardware]] |
