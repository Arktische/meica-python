from __future__ import annotations
from typing import Dict

GLOBAL_STEP = "global_step"
TRAINED_SAMPLES_COUNT = "trained_samples_count"
NUM_PROCESSES = "num_processes"
EPOCH = "epoch"
STEP = "step"
ITERATION = "iteration"


class ProgressTracker:
    def __init__(self, trainer):
        self.__global_step__ = 0
        self.__epoch__ = 0
        self.__step__ = 0
        self.__trainer__ = trainer

    def state_dict(self) -> Dict:
        return {
            GLOBAL_STEP: self.__global_step__,
            NUM_PROCESSES: self.__trainer__.num_processes,
            EPOCH: self.__epoch__,
            STEP: self.__step__,
            ITERATION: self.__trainer__.project_configuration.iteration,
        }

    def load_state_dict(self, state: Dict) -> None:
        num_processes = state.get(NUM_PROCESSES, None)
        global_step = state.get(GLOBAL_STEP, None)
        epoch = state.get(EPOCH, None)
        step = state.get(STEP, None)
        iteration = state.get(ITERATION, None)
        if (
            num_processes is None
            or global_step is None
            or epoch is None
            or step is None
            or iteration is None
        ):
            raise ValueError(
                f"progress state dict must contain {NUM_PROCESSES}, {GLOBAL_STEP}, {EPOCH}, {STEP}, {ITERATION}, but got {state.keys()}"
            )

        if num_processes != self.__trainer__.num_processes:
            raise ValueError(
                f"num_processes mismatch so we can't resume training, expect num_processes {self.__trainer__.num_processes} from config, but got {num_processes} from checkpoint, do you change the num_processes?"
            )
        if global_step < 0 or step < 0:
            raise ValueError(
                f"global_step and step must be greater than 0, but got global_step={global_step} and step={step} from checkpoint"
            )
        self.__global_step__ = global_step
        self.__epoch__ = epoch
        self.__step__ = step
        self.__trainer__.project_configuration.iteration = iteration

    @property
    def global_step(self) -> int:
        return self.__global_step__

    @property
    def epoch(self) -> int:
        return self.__epoch__

    @property
    def step(self) -> int:
        return self.__step__

    def update(self, epoch, step):
        self.__epoch__ = epoch
        self.__step__ = step
        self.__global_step__ += 1
