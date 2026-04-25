"""LiNexa-enabled Psi0 server entrypoint.

Imports the upstream ``psi.deploy.psi0_serve_simple``, monkey-patches its
``Server`` class so that:

  * after the model is loaded, :func:`linexa.adapters.psi0.install` wraps the
    selected action-expert ``ff_act`` modules
  * each ``predict_action`` call peeks at ``payload["history"]`` for the
    existing ``"reset"`` signal and clears the fast-weight cache when present

then delegates to the upstream ``main()`` so all CLI handling, tyro config
parsing, and FastAPI/uvicorn wiring stay in upstream code. This file does not
modify any submodule source.

Activated by ``LINEXA_TTT_ENABLED=1`` in the docker entrypoint script. When
disabled, the upstream ``serve_psi0`` console script is exec'd directly and
this module is never imported.
"""
from __future__ import annotations

import logging
import sys

from linexa.adapters import psi0 as adapter
from linexa.ttt.config import LinexaConfig

logger = logging.getLogger(__name__)


def _patch(upstream) -> None:
    OriginalServer = upstream.Server

    class LinexaServer(OriginalServer):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._linexa_cfg = LinexaConfig.from_env()
            adapter.install(self.model, self._linexa_cfg)

        def predict_action(self, payload):
            if isinstance(payload, dict):
                history = payload.get("history")
                if isinstance(history, dict) and "reset" in history:
                    adapter.reset(self.model)
            return super().predict_action(payload)

    upstream.Server = LinexaServer
    logger.info("linexa: monkey-patched %s.Server -> LinexaServer", upstream.__name__)


def main() -> None:
    from psi.deploy import psi0_serve_simple as upstream

    _patch(upstream)
    upstream.main()


if __name__ == "__main__":
    sys.exit(main() or 0)
