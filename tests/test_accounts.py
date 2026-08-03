import numpy as np
import pytest

from lifecycle_finance import AccountEventResult, AccountType, TaxPolicy
from lifecycle_finance.accounts import (
    Account,
    AccountPortfolio,
    AccountProjection,
    Asset,
    Contribution,
    RothConversion,
    TaxLedger,
    TaxLot,
    Withdrawal,
    rebalance_asset_location,
)


@pytest.fixture
def assets() -> dict[str, Asset]:
    return {
        "STOCK": Asset("STOCK", "equity", 0.05, 0.015, 1.0),
        "BOND": Asset("BOND", "bonds", 0.02, 0.02, 0.0),
    }


def test_hifo_sale_tracks_basis_and_gain() -> None:
    account = Account(
        "taxable",
        AccountType.TAXABLE,
        [
            TaxLot("STOCK", 10, 500, 2020),
            TaxLot("STOCK", 10, 900, 2021),
        ],
    )
    proceeds, gain = account.sell(
        "STOCK",
        1_000,
        100,
        sale_year=2026,
        method="hifo",
    )
    assert proceeds == pytest.approx(1_000)
    assert gain == pytest.approx(100)
    assert account.market_value({"STOCK": 100}) == pytest.approx(1_000)


def test_fifo_sale_uses_acquisition_order() -> None:
    account = Account(
        "taxable",
        AccountType.TAXABLE,
        [
            TaxLot("STOCK", 5, 200, 2020),
            TaxLot("STOCK", 5, 450, 2025),
        ],
    )

    proceeds, gain = account.sell(
        "STOCK",
        500,
        100,
        method="fifo",
        sale_year=2026,
    )

    assert proceeds == pytest.approx(500)
    assert gain == pytest.approx(300)
    assert account.lots == [TaxLot("STOCK", 5, 450, 2025)]


def test_conversion_and_withdrawal_record_correct_income() -> None:
    portfolio = AccountPortfolio(
        {
            "traditional": Account(
                "traditional",
                AccountType.TRADITIONAL,
                [TaxLot("BOND", 20, 2_000, 2020)],
            ),
            "roth": Account("roth", AccountType.ROTH),
        },
        {"STOCK": 100, "BOND": 100},
    )
    ledger = TaxLedger()
    RothConversion("traditional", "roth", 500, "BOND").apply(portfolio, 2026, ledger)
    assert ledger.ordinary_income == pytest.approx(500)
    assert portfolio.accounts["roth"].market_value(portfolio.prices) == pytest.approx(500)

    Withdrawal("traditional", 250).apply(portfolio, 2026, ledger)
    assert ledger.ordinary_income == pytest.approx(750)


def test_roth_conversion_reports_an_execution_shortfall() -> None:
    portfolio = AccountPortfolio(
        {
            "traditional": Account(
                "traditional",
                AccountType.TRADITIONAL,
                [TaxLot("BOND", 10, 1_000, 2020)],
            ),
            "roth": Account("roth", AccountType.ROTH),
        },
        {"BOND": 100},
    )
    ledger = TaxLedger()

    result = RothConversion("traditional", "roth", 1_500, "BOND").apply(
        portfolio,
        2026,
        ledger,
    )

    assert result.requested == 1_500
    assert result.executed == pytest.approx(1_000)
    assert result.shortfall == pytest.approx(500)
    assert isinstance(result, AccountEventResult)


def test_withdrawal_reports_an_execution_shortfall() -> None:
    portfolio = AccountPortfolio(
        {
            "traditional": Account(
                "traditional",
                AccountType.TRADITIONAL,
                [TaxLot("BOND", 10, 1_000, 2020)],
            ),
        },
        {"BOND": 100},
    )

    result = Withdrawal("traditional", 1_500).apply(portfolio, 2026, TaxLedger())

    assert result.requested == 1_500
    assert result.executed == pytest.approx(1_000)
    assert result.shortfall == pytest.approx(500)


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_withdrawal_amount_preserves_portfolio_and_ledger(
    invalid: float,
) -> None:
    portfolio = AccountPortfolio(
        {
            "traditional": Account(
                "traditional",
                AccountType.TRADITIONAL,
                [TaxLot("BOND", 10, 1_000, 2020)],
            ),
        },
        {"BOND": 100},
    )
    ledger = TaxLedger(ordinary_income=25)
    before = portfolio.clone()

    with pytest.raises(ValueError, match="amount"):
        Withdrawal("traditional", invalid).apply(portfolio, 2026, ledger)

    assert portfolio == before
    assert ledger == TaxLedger(ordinary_income=25)


def test_failed_roth_conversion_preserves_portfolio_and_ledger() -> None:
    portfolio = AccountPortfolio(
        {
            "traditional": Account(
                "traditional",
                AccountType.TRADITIONAL,
                [TaxLot("BOND", 10, 1_000, 2020)],
            ),
            "roth": Account(
                "roth",
                AccountType.ROTH,
                allowed_assets=frozenset({"STOCK"}),
            ),
        },
        {"BOND": 100, "STOCK": 100},
    )
    ledger = TaxLedger(ordinary_income=25)
    before = portfolio.clone()

    with pytest.raises(ValueError, match="not allowed"):
        RothConversion("traditional", "roth", 500, "BOND").apply(
            portfolio,
            2026,
            ledger,
        )

    assert portfolio == before
    assert ledger == TaxLedger(ordinary_income=25)


def test_public_sales_and_liquidations_accept_a_sale_year() -> None:
    sale_account = Account(
        "taxable",
        AccountType.TAXABLE,
        [TaxLot("STOCK", 10, 500, 2026)],
    )
    liquidation_account = Account(
        "taxable",
        AccountType.TAXABLE,
        [TaxLot("STOCK", 10, 500, 2026)],
    )

    assert sale_account.sell("STOCK", 1_000, 100, sale_year=2026) == pytest.approx(
        (1_000, 500)
    )
    assert liquidation_account.liquidate(
        {"STOCK": 100},
        sale_year=2026,
    ) == pytest.approx((1_000, 500))


def test_public_sales_and_liquidations_require_a_sale_year() -> None:
    account = Account(
        "taxable",
        AccountType.TAXABLE,
        [TaxLot("STOCK", 10, 500, 2020)],
    )

    with pytest.raises(TypeError):
        account.sell("STOCK", 1_000, 100)
    with pytest.raises(TypeError):
        account.liquidate({"STOCK": 100})


def test_asset_location_requires_an_explicit_year() -> None:
    with pytest.raises(TypeError):
        rebalance_asset_location(AccountPortfolio({}, {}), {}, {})


def test_asset_location_application_failure_is_transactional() -> None:
    assets = {
        "STOCK": Asset("STOCK", "equity", 0.05),
        "BOND": Asset("BOND", "bonds", 0.02),
    }
    portfolio = AccountPortfolio(
        {
            "taxable": Account(
                "taxable",
                AccountType.TAXABLE,
                [TaxLot("STOCK", 10, 500, 2020)],
            ),
        },
        {"STOCK": 100},
    )
    ledger = TaxLedger(ordinary_income=25)
    before = portfolio.clone()

    with pytest.raises(KeyError, match="BOND"):
        rebalance_asset_location(
            portfolio,
            assets,
            {"equity": 0.0, "bonds": 1.0},
            year=2026,
            ledger=ledger,
        )

    assert portfolio == before
    assert ledger == TaxLedger(ordinary_income=25)


def test_asset_location_preserves_account_budgets(
    assets: dict[str, Asset],
) -> None:
    portfolio = AccountPortfolio(
        {
            "traditional": Account(
                "traditional",
                AccountType.TRADITIONAL,
                [TaxLot("STOCK", 10, 1_000, 2020)],
                frozenset(assets),
            ),
            "roth": Account(
                "roth",
                AccountType.ROTH,
                [TaxLot("BOND", 10, 1_000, 2020)],
                frozenset(assets),
            ),
        },
        {"STOCK": 100, "BOND": 100},
    )
    rebalance_asset_location(
        portfolio,
        assets,
        {"equity": 0.6, "bonds": 0.4},
        year=2026,
    )
    weights = portfolio.bucket_weights(assets)
    assert portfolio.market_value() == pytest.approx(2_000)
    assert weights["equity"] == pytest.approx(0.6)
    assert weights["bonds"] == pytest.approx(0.4)
    assert portfolio.accounts["traditional"].market_value(portfolio.prices) == pytest.approx(1_000)
    assert portfolio.accounts["roth"].market_value(portfolio.prices) == pytest.approx(1_000)


def test_asset_location_noop_preserves_tax_lots_and_realizes_no_gain(
    assets: dict[str, Asset],
) -> None:
    lot = TaxLot("STOCK", 3_000, 100_000, 2020)
    portfolio = AccountPortfolio(
        {
            "taxable": Account(
                "taxable",
                AccountType.TAXABLE,
                [lot],
                frozenset(assets),
            ),
        },
        {"STOCK": 100, "BOND": 100},
    )
    ledger = TaxLedger()

    rebalance_asset_location(
        portfolio,
        assets,
        {"equity": 1.0, "bonds": 0.0},
        year=2026,
        ledger=ledger,
    )

    assert portfolio.accounts["taxable"].lots == [lot]
    assert lot.shares == 3_000
    assert lot.basis == 100_000
    assert ledger.long_term_capital_gains == 0.0


def test_asset_location_noop_preserves_symbol_within_target_bucket() -> None:
    equivalent_assets = {
        "A": Asset("A", "equity", 0.05),
        "B": Asset("B", "equity", 0.05),
    }
    lot = TaxLot("B", 1_000, 25_000, 2020)
    portfolio = AccountPortfolio(
        {
            "taxable": Account(
                "taxable",
                AccountType.TAXABLE,
                [lot],
                frozenset(equivalent_assets),
            ),
        },
        {"A": 100, "B": 100},
    )
    ledger = TaxLedger()

    allocations = rebalance_asset_location(
        portfolio,
        equivalent_assets,
        {"equity": 1.0},
        year=2026,
        ledger=ledger,
    )

    assert allocations == {("taxable", "B"): 100_000}
    assert portfolio.accounts["taxable"].lots == [lot]
    assert lot.shares == 1_000
    assert lot.basis == 25_000
    assert ledger.long_term_capital_gains == 0.0


def test_asset_location_noop_preserves_target_buckets_across_accounts(
    assets: dict[str, Asset],
) -> None:
    taxable_lot = TaxLot("BOND", 10, 500, 2020)
    roth_lot = TaxLot("STOCK", 10, 1_000, 2020)
    portfolio = AccountPortfolio(
        {
            "taxable": Account("taxable", AccountType.TAXABLE, [taxable_lot]),
            "roth": Account("roth", AccountType.ROTH, [roth_lot]),
        },
        {"STOCK": 100, "BOND": 100},
    )
    ledger = TaxLedger()

    allocations = rebalance_asset_location(
        portfolio,
        assets,
        {"equity": 0.5, "bonds": 0.5},
        year=2026,
        ledger=ledger,
    )

    assert allocations == {
        ("taxable", "BOND"): 1_000,
        ("roth", "STOCK"): 1_000,
    }
    assert portfolio.accounts["taxable"].lots == [taxable_lot]
    assert portfolio.accounts["roth"].lots == [roth_lot]
    assert ledger.long_term_capital_gains == 0


def test_asset_location_prefers_required_turnover_in_tax_advantaged_account(
    assets: dict[str, Asset],
) -> None:
    portfolio = AccountPortfolio(
        {
            "roth": Account(
                "roth",
                AccountType.ROTH,
                [
                    TaxLot("STOCK", 5, 500, 2020),
                    TaxLot("BOND", 5, 500, 2020),
                ],
            ),
            "taxable": Account(
                "taxable",
                AccountType.TAXABLE,
                [
                    TaxLot("STOCK", 5, 100, 2020),
                    TaxLot("BOND", 5, 100, 2020),
                ],
            ),
        },
        {"STOCK": 100, "BOND": 100},
    )
    ledger = TaxLedger()

    allocations = rebalance_asset_location(
        portfolio,
        assets,
        {"equity": 0.6, "bonds": 0.4},
        year=2026,
        ledger=ledger,
    )

    assert allocations == pytest.approx(
        {
            ("roth", "STOCK"): 700,
            ("roth", "BOND"): 300,
            ("taxable", "STOCK"): 500,
            ("taxable", "BOND"): 500,
        }
    )
    assert ledger.long_term_capital_gains == 0


def test_asset_location_realizes_gain_only_on_the_required_delta(
    assets: dict[str, Asset],
) -> None:
    portfolio = AccountPortfolio(
        {
            "taxable": Account(
                "taxable",
                AccountType.TAXABLE,
                [TaxLot("STOCK", 1_000, 50_000, 2020)],
                frozenset(assets),
            ),
        },
        {"STOCK": 100, "BOND": 100},
    )
    ledger = TaxLedger()

    rebalance_asset_location(
        portfolio,
        assets,
        {"equity": 0.5, "bonds": 0.5},
        year=2026,
        ledger=ledger,
    )

    stock_lot = next(
        lot for lot in portfolio.accounts["taxable"].lots if lot.asset == "STOCK"
    )
    assert stock_lot.shares == pytest.approx(500)
    assert stock_lot.basis == pytest.approx(25_000)
    assert ledger.long_term_capital_gains == pytest.approx(25_000)


def test_taxable_withdrawal_distinguishes_short_and_long_term_gains() -> None:
    portfolio = AccountPortfolio(
        {
            "short": Account(
                "short",
                AccountType.TAXABLE,
                [TaxLot("STOCK", 10, 500, 2026)],
            ),
            "long": Account(
                "long",
                AccountType.TAXABLE,
                [TaxLot("STOCK", 10, 500, 2025)],
            ),
        },
        {"STOCK": 100},
    )
    ledger = TaxLedger()

    Withdrawal("short", 1_000).apply(portfolio, 2026, ledger)
    Withdrawal("long", 1_000).apply(portfolio, 2026, ledger)

    assert ledger.short_term_capital_gains == pytest.approx(500)
    assert ledger.long_term_capital_gains == pytest.approx(500)


def test_nonqualified_hsa_withdrawal_records_income_and_penalty() -> None:
    portfolio = AccountPortfolio(
        {
            "hsa": Account(
                "hsa",
                AccountType.HSA,
                [TaxLot("BOND", 10, 1_000, 2020)],
            ),
        },
        {"BOND": 100},
    )
    ledger = TaxLedger()

    result = Withdrawal("hsa", 250, qualified_hsa=False).apply(
        portfolio,
        2026,
        ledger,
    )

    assert result.executed == pytest.approx(250)
    assert ledger.ordinary_income == pytest.approx(250)
    assert ledger.hsa_penalties == pytest.approx(50)


def test_tax_ledger_preserves_original_positional_field_order() -> None:
    ledger = TaxLedger(1, 2, 3, 4, 5, 6, 7)

    assert ledger.wages == 1
    assert ledger.ordinary_income == 2
    assert ledger.long_term_capital_gains == 3
    assert ledger.qualified_dividends == 4
    assert ledger.interest == 5
    assert ledger.social_security == 6
    assert ledger.hsa_penalties == 7
    assert ledger.short_term_capital_gains == 0
    assert ledger.nonqualified_dividends == 0


def test_capital_loss_deduction_and_carryforward_offset_a_later_gain() -> None:
    brackets = ((float("inf"), 0.10),)
    policy = TaxPolicy(
        ordinary_brackets=brackets,
        capital_gains_brackets=brackets,
        standard_deduction=0.0,
        state_ordinary_rate=0.0,
        niit_rate=0.0,
    )
    loss_year = TaxLedger(
        wages=10_000,
        short_term_capital_gains=-5_000,
    ).tax(policy)
    gain_year = TaxLedger(short_term_capital_gains=5_000).tax(
        policy,
        short_term_loss_carryforward=loss_year.short_term_loss_carryforward,
    )

    assert loss_year.capital_loss_deduction == 3_000
    assert loss_year.short_term_loss_carryforward == 2_000
    assert loss_year.federal_ordinary == pytest.approx(700)
    assert gain_year.short_term_loss_carryforward == 0
    assert gain_year.federal_ordinary == pytest.approx(300)


def test_capital_loss_carryforward_is_not_consumed_without_taxable_income() -> None:
    result = TaxLedger(short_term_capital_gains=-5_000).tax(TaxPolicy.for_2026())

    assert result.capital_loss_deduction == 0
    assert result.short_term_loss_carryforward == 5_000


def test_account_projection_rolls_capital_losses_into_the_next_year(
    assets: dict[str, Asset],
) -> None:
    class CapitalGainEvent:
        def __init__(self, amount: float) -> None:
            self.amount = amount

        def apply(
            self,
            portfolio: AccountPortfolio,
            year: int,
            ledger: TaxLedger,
        ) -> None:
            ledger.long_term_capital_gains += self.amount

    brackets = ((float("inf"), 0.10),)
    policy = TaxPolicy(
        ordinary_brackets=brackets,
        capital_gains_brackets=brackets,
        standard_deduction=0.0,
        state_ordinary_rate=0.0,
        niit_rate=0.0,
    )
    projection = AccountProjection(assets, policy)
    initial = AccountPortfolio({}, {"STOCK": 100, "BOND": 100})

    _, history = projection.run(
        initial,
        start_year=2026,
        end_year=2027,
        events={
            2026: [CapitalGainEvent(-5_000)],
            2027: [CapitalGainEvent(5_000)],
        },
    )

    assert history[0].taxes.long_term_loss_carryforward == 5_000
    assert history[1].taxes.federal_capital_gains == 0


def test_nonqualified_dividends_are_included_in_niit() -> None:
    brackets = ((float("inf"), 0.0),)
    policy = TaxPolicy(
        ordinary_brackets=brackets,
        capital_gains_brackets=brackets,
        standard_deduction=0.0,
        state_ordinary_rate=0.0,
        niit_threshold=0.0,
    )

    result = TaxLedger(nonqualified_dividends=1_000).tax(policy)

    assert result.net_investment_income_tax == pytest.approx(38.0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_tax_lots_reject_nonfinite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        TaxLot("STOCK", invalid, 100.0, 2020)
    with pytest.raises(ValueError, match="finite"):
        TaxLot("STOCK", 1.0, invalid, 2020)
    with pytest.raises(ValueError, match="finite"):
        TaxLot("STOCK", 1.0, 100.0, invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_account_trades_reject_nonfinite_values(invalid: float) -> None:
    account = Account("taxable", AccountType.TAXABLE)
    with pytest.raises(ValueError, match="finite"):
        account.buy("STOCK", invalid, 100.0, 2026)
    with pytest.raises(ValueError, match="finite"):
        account.buy("STOCK", 100.0, invalid, 2026)
    with pytest.raises(ValueError, match="finite"):
        account.buy("STOCK", 100.0, 100.0, invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        account.sell("STOCK", invalid, 100.0, sale_year=2026)
    with pytest.raises(ValueError, match="finite"):
        account.sell("STOCK", 100.0, invalid, sale_year=2026)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_account_financial_boundaries_reject_nonfinite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Asset("STOCK", "equity", invalid)
    with pytest.raises(ValueError, match="finite"):
        Asset("STOCK", "equity", 0.05, invalid)
    with pytest.raises(ValueError, match="finite"):
        AccountPortfolio({}, {"STOCK": invalid})


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_tax_ledger_rejects_nonfinite_hsa_penalties(invalid: float) -> None:
    with pytest.raises(ValueError, match="hsa_penalties"):
        TaxLedger(hsa_penalties=invalid).tax(TaxPolicy.for_2026())


def test_account_projection_runs_events_and_tax_ledgers(
    assets: dict[str, Asset],
) -> None:
    initial = AccountPortfolio(
        {
            "taxable": Account(
                "taxable",
                AccountType.TAXABLE,
                [TaxLot("STOCK", 10, 1_000, 2025)],
            )
        },
        {"STOCK": 100, "BOND": 100},
    )
    projection = AccountProjection(assets, TaxPolicy.for_2026())
    final, history = projection.run(
        initial,
        start_year=2026,
        end_year=2027,
        events={2026: [Contribution("taxable", "BOND", 500)]},
    )
    assert len(history) == 2
    assert final.market_value() > initial.market_value()
    assert history[0].taxes.total > 0
