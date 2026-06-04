# SPDX-License-Identifier: Apache-2.0
global_output_simulator = None


def set_global_output_simulator(controller):
    global global_output_simulator
    global_output_simulator = controller


def get_global_output_simulator():
    global global_output_simulator
    if global_output_simulator is None:
        raise RuntimeError("OutputSimulator has not been initialized yet.")
    return global_output_simulator
