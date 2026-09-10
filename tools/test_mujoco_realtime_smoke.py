#!/usr/bin/env python3
"""Run isolated, hardware-free MuJoCo pacing/configuration/shutdown smoke tests.

Run after building and sourcing the workspace:
  python3 tools/test_mujoco_realtime_smoke.py
No robot plugins are loaded. Logs are kept under --output-dir.
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=Path, help="Override installed mujoco_node executable")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/mujoco-realtime-smoke"))
    parser.add_argument("--domain-id", type=int, default=175)
    args = parser.parse_args()
    if args.node is None:
        prefix = subprocess.check_output(["ros2", "pkg", "prefix", "mujoco_ros"], text=True).strip()
        args.node = Path(prefix) / "lib/mujoco_ros/mujoco_node"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = Path(__file__).resolve().parents[1] / "mujoco_ros_pkgs/mujoco_ros/assets/pendulum_world.xml"
    env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain_id), ROS_LOCALHOST_ONLY="1",
               ROS_LOG_DIR=str(args.output_dir / "ros_logs"))
    cases = {
        "paced": {"physics_wall_pacing": "true"},
        "legacy": {"physics_wall_pacing": "false"},
        "unlimited": {"physics_wall_pacing": "true", "realtime": "-1.0"},
        "no_render_override": {"headless": "false", "render_offscreen": "true", "no_render": "true"},
    }
    for name, overrides in cases.items():
        timing = args.output_dir / f"{name}.csv"
        params = {"use_sim_time": "true", "headless": "true", "render_offscreen": "false",
                  "unpause": "true", "num_steps": "1000", "modelfile": str(model),
                  "physics_timing_path": str(timing)}
        params.update(overrides)
        command = [str(args.node), "--ros-args"]
        for key, value in params.items():
            command += ["-p", f"{key}:={value}"]
        with (args.output_dir / f"{name}.log").open("w") as output:
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=20, check=True)
        with timing.open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["channel"] == "physics"]
        assert sum(int(r["steps"]) for r in rows) == 1000, f"{name}: lost/extra steps or trace records"
        assert abs(float(rows[-1]["sim_time"]) - 1.0) < 1e-8, f"{name}: incorrect simulation advance"
        assert all(int(r["dropped_total"]) == 0 for r in rows), f"{name}: trace overflow"
        if name in ("paced", "no_render_override"):
            assert all(int(r["steps"]) == 1 for r in rows), f"{name}: unexpected catch-up batch"
        log = (args.output_dir / f"{name}.log").read_text()
        assert "headless=1 offscreen=0" in log, f"{name}: effective render settings incorrect"
        print(f"PASS {name}: 1000 steps, 1.0 s simulation, clean automatic exit", flush=True)
    print(f"Logs: {args.output_dir}")


if __name__ == "__main__":
    main()
