from stable_baselines.common.callbacks import BaseCallback, EvalCallback
import os
from datetime import datetime

from exp.utils.path_utils import save_json_to_file


class NewBestCallback(BaseCallback):
    """
    Callback to be invoked by EvalCallback on callback_on_new_best event
    """

    def __init__(self, log_dir, verbose=1):
        super(NewBestCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = log_dir
        self.index = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.parent is not None:
            evaluations_results = self.parent.evaluations_results
            evaluations_length = self.parent.evaluations_length
        elif self.locals["callback"] is not None and isinstance(
            self.locals["callback"].callbacks[0], EvalCallback
        ):
            evaluations_results = (
                self.locals["callback"].callbacks[0].evaluations_results
            )
            evaluations_length = self.locals["callback"].callbacks[0].evaluations_length
        else:
            raise ValueError("can't get values")

        d = {
            "evaluations_results": evaluations_results,
            "evaluations_length": evaluations_length,
        }
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_json_to_file(
            report_save_path=self.save_path,
            file_name=f"eval_{self.index}_{now}.txt",
            **d,
        )

        self.index += 1

        return True
