"""fontdiffuser training entry — Phase 1 reimpl-worker must implement main()."""
from __future__ import annotations


def main(args, *, data_cfg, model_cfg, train_cfg, paths):
    """Called by paper_reimpl_shared.runner.entrypoint.

    Phase 1 worker: implement model build + dataset + training loop here.
    Must respect args.dry_run and args.synthetic flags.
    """
    raise NotImplementedError(
        "fontdiffuser.train.main() not implemented yet. "
        "Phase 1 reimpl-worker must fill this in."
    )
