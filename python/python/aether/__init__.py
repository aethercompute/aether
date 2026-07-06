__all__ = [
    "DistroResult",
    "PretrainedSourceRepoFiles",
    "PretrainedSourceStateDict",
    "Trainer",
    "make_causal_lm",
    "start_process_watcher",
]


def __getattr__(name):
    if name in {
        "make_causal_lm",
        "PretrainedSourceRepoFiles",
        "PretrainedSourceStateDict",
    }:
        from .models import (
            PretrainedSourceRepoFiles,
            PretrainedSourceStateDict,
            make_causal_lm,
        )

        return {
            "make_causal_lm": make_causal_lm,
            "PretrainedSourceRepoFiles": PretrainedSourceRepoFiles,
            "PretrainedSourceStateDict": PretrainedSourceStateDict,
        }[name]

    if name in {"Trainer", "DistroResult", "start_process_watcher"}:
        from ._ext import DistroResult, Trainer, start_process_watcher

        return {
            "Trainer": Trainer,
            "DistroResult": DistroResult,
            "start_process_watcher": start_process_watcher,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
