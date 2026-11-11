from marketpulse.generation.factcheck import fact_check
from marketpulse.generation.facts import DailyFacts, Move, NewsItem


def make_facts():
    return DailyFacts(
        date="2024-08-05", regime="stressed", regime_probability=87, previous_regime="normal",
        moves=[
            Move(instrument_id="SPX", name="S&P 500", kind="price", unit="%", level=5186.33,
                 change=-3.04, z=-2.9, flagged=True),
            Move(instrument_id="UST2Y", name="US 2Y yield", kind="rate", unit="bp", level=3.88,
                 change=-10.4, z=-1.6, flagged=False),
            Move(instrument_id="NIFTY", name="Nifty 50", kind="price", unit="%", level=24055.6,
                 change=-2.68, z=-2.7, flagged=True),
        ],
        news={"SPX": [NewsItem(title="Fed seen cutting by 50bp after weak jobs report", source="x")]},
    )


def test_correct_brief_passes(cfg):
    text = ("Markets are in a stressed regime (87% probability), switching from the normal regime. "
            "The S&P 500 fell 3.04% to 5,186.33, a 2.9 standard deviation move. "
            "Traders now expect the Fed to cut by 50bp. "
            "The Nifty 50 dropped 2.68% to 24,055.60. The US 2Y yield eased 10.4bp to 3.88%.")
    r = fact_check(text, make_facts(), cfg)
    assert r.passed, r.feedback()
    assert r.numbers_checked >= 8  # names like '500', '50', '2Y' are not counted as data


def test_invented_number_is_caught(cfg):
    r = fact_check("The S&P 500 fell 3.40% on the day.", make_facts(), cfg)
    assert r.unsupported_numbers == ["3.40"]


def test_number_given_to_wrong_market_is_caught(cfg):
    # 2.68 is a real number, but it is the Nifty's move, not the S&P 500's
    r = fact_check("The S&P 500 fell 2.68%.", make_facts(), cfg)
    assert not r.passed


def test_rounding_is_allowed(cfg):
    assert fact_check("The S&P 500 fell about 3.0%.", make_facts(), cfg).passed
    assert fact_check("The S&P 500 fell about 3%.", make_facts(), cfg).passed


def test_wrong_direction_is_caught(cfg):
    r = fact_check("The S&P 500 rallied 3.04%.", make_facts(), cfg)
    assert r.direction_errors


def test_wrong_regime_is_caught(cfg):
    r = fact_check("Markets are in a calm regime.", make_facts(), cfg)
    assert r.regime_errors


def test_date_is_not_checked_as_data(cfg):
    assert fact_check("On 5 August 2024 the S&P 500 fell 3.04%.", make_facts(), cfg).passed
