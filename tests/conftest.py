import logging

import pytest

from marketpulse.config import load_config
from marketpulse.data.market import compute_changes
from marketpulse.data.synthetic import generate

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def demo(cfg):
    levels, news, truth = generate(cfg, n_days=1600, n_shocks=30, seed=3)
    return {"levels": levels, "news": news, "truth": truth, "changes": compute_changes(levels, cfg)}
