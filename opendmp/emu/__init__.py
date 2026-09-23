"""Timed, bit-exact emulator of the InvenSense DMP (MPU-6050 and ICM-20948).

    from opendmp.emu import Machine, Sample, run, read_sensor_csv
    m = Machine("mpu6050")
    m.load(code, 0x400)
    results = run(m, read_sensor_csv("walk.csv"), rate_hz=200)

Command line: `python -m opendmp.emu --help`; synthetic sensor data: `python -m opendmp.emu.gen`.
"""
from .chips import CHIPS, ICM20948, MPU6050, Chip, get_chip
from .files import read_program, read_sensor_csv, write_sensor_csv
from .machine import DmpFault, Machine, Model, PassResult, Sample, Unmodelled
from .sched import run, summary

__all__ = ["CHIPS", "ICM20948", "MPU6050", "Chip", "get_chip", "read_program", "read_sensor_csv",
           "write_sensor_csv", "DmpFault", "Machine", "Model", "PassResult", "Sample",
           "Unmodelled", "run", "summary"]
