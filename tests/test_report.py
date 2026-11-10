import numpy as np
import pandas as pd
import pytest

import report
from matplotlib import image as mpl_image
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def equity(values):
    return pd.Series(values, index=pd.date_range("2020-12-31", periods=len(values), tz="UTC"), dtype=float)


@pytest.fixture
def saved_figures(monkeypatch):
    # Keep real rendering/file IO; retain the closed figure to inspect plotted data.
    figures = []
    savefig = Figure.savefig

    def save_and_capture(figure, *args, **kwargs):
        savefig(figure, *args, **kwargs)
        figures.append(figure)

    monkeypatch.setattr(Figure, "savefig", save_and_capture)
    yield figures
    for figure in figures:
        plt.close(figure)


def assert_readable_png(path):
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    pixels = mpl_image.imread(path)
    assert pixels.ndim == 3
    assert pixels.shape[0] >= 300
    assert pixels.shape[1] >= 600
    assert np.isfinite(pixels).all()
    assert pixels.std() > 0


@pytest.mark.parametrize("boundary", ["2021-01-01", pd.Timestamp("2021-01-01", tz="UTC")])
def test_equity_chart_normalizes_and_shows_aligned_drawdowns(tmp_path, saved_figures, boundary):
    nav = equity([100, 90, 99])
    original = nav.copy(deep=True)
    path = tmp_path / "equity.png"
    report.equity_chart({"Strategy": nav, "Cash": equity([500, 500, 500])}, path,
                        "Research comparison", vline=boundary)
    assert_readable_png(path)
    figure = saved_figures[-1]
    assert len(figure.axes) == 2
    top, bottom = figure.axes
    assert top.get_shared_x_axes().joined(top, bottom)
    assert top.get_yscale() == "log"
    assert top.get_title() == "Research comparison"
    nav_lines = {line.get_label(): line for line in top.lines}
    dd_lines = {line.get_label(): line for line in bottom.lines}
    assert nav_lines["Strategy"].get_ydata() == pytest.approx([1, 0.9, 0.99])
    assert nav_lines["Cash"].get_ydata() == pytest.approx([1, 1, 1])
    assert dd_lines["Strategy"].get_ydata() == pytest.approx([0, -0.1, -0.01])
    assert dd_lines["Cash"].get_ydata() == pytest.approx([0, 0, 0])
    assert nav_lines["Strategy"].get_color() == dd_lines["Strategy"].get_color()
    expected_label = "historical validation (previously inspected)"
    legend_labels = [text.get_text() for text in top.get_legend().get_texts()]
    assert expected_label in legend_labels
    for axis in (top, bottom):
        vertical = [line for line in axis.lines if line.get_linestyle() == "--"]
        assert len(vertical) == 1
        assert pd.Timestamp(vertical[0].get_xdata()[0]) == pd.Timestamp("2021-01-01", tz="UTC")
    assert not plt.fignum_exists(figure.number)
    assert report.matplotlib.get_backend().lower() == "agg"
    pd.testing.assert_series_equal(nav, original)


def test_equity_chart_nonpositive_nav_uses_linear_axis(tmp_path, saved_figures):
    path = tmp_path / "depleted.png"
    report.equity_chart({"Depleted": equity([100, 0, -10])}, path, "Depletion")
    assert_readable_png(path)
    top, bottom = saved_figures[-1].axes
    assert top.get_yscale() == "linear"
    assert top.lines[0].get_ydata() == pytest.approx([1, 0, -0.1])
    assert bottom.lines[0].get_ydata() == pytest.approx([0, -1, -1.1])


def test_equity_chart_does_not_truncate_general_series_map(tmp_path, saved_figures):
    path = tmp_path / "many.png"
    series = {f"Series {i}": equity([100, 100 + i, 100]) for i in range(7)}
    report.equity_chart(series, path, "Comparison")
    assert_readable_png(path)
    for axis in saved_figures[-1].axes:
        labels = {line.get_label() for line in axis.lines}
        assert set(series).issubset(labels)


@pytest.mark.parametrize("series", [
    {}, {"Empty": equity([])}, {"Zero start": equity([0, 1])},
    {"Missing": equity([100, np.nan])}, {"Infinite": equity([100, np.inf])},
])
def test_equity_chart_rejects_unplottable_series(tmp_path, series):
    with pytest.raises(ValueError):
        report.equity_chart(series, tmp_path / "bad.png", "Invalid")


@pytest.mark.parametrize("boundary", ["not-a-date", pd.NaT])
def test_equity_chart_rejects_invalid_boundary(tmp_path, boundary):
    with pytest.raises(ValueError):
        report.equity_chart({"Cash": equity([100, 100])}, tmp_path / "bad.png", "Invalid", boundary)


def test_exposure_chart_uses_realized_long_absolute_short_and_gross(tmp_path, saved_figures):
    weights = pd.DataFrame({"BTC": [0.6, -0.2, 0, 1.25], "ETH": [-0.4, -0.3, 0, 0.25]},
                           index=equity([100] * 4).index)
    original = weights.copy(deep=True)
    path = tmp_path / "exposure.png"
    report.exposure_chart(weights, path, "Realized exposure")
    assert_readable_png(path)
    figure = saved_figures[-1]
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    lines = {line.get_label().lower(): line for line in axis.lines}
    assert lines["long"].get_ydata() == pytest.approx([0.6, 0, 0, 1.5])
    assert lines["short"].get_ydata() == pytest.approx([0.4, 0.5, 0, 0])
    assert lines["gross"].get_ydata() == pytest.approx([1, 0.5, 0, 1.5])
    assert list(lines["long"].get_xdata()) == list(weights.index)
    assert any(np.equal(line.get_ydata(), 0).all() for line in axis.lines)
    assert "realized" in axis.get_ylabel().lower()
    assert "eod" in axis.get_ylabel().lower()
    assert axis.get_title() == "Realized exposure"
    assert not plt.fignum_exists(figure.number)
    pd.testing.assert_frame_equal(weights, original)


@pytest.mark.parametrize("weights", [
    pd.DataFrame(),
    pd.DataFrame({"BTC": [0, np.nan]}, index=equity([100, 100]).index),
    pd.DataFrame({"BTC": [0, np.inf]}, index=equity([100, 100]).index),
])
def test_exposure_chart_rejects_missing_exposure_instead_of_assuming_zero(tmp_path, weights):
    with pytest.raises(ValueError):
        report.exposure_chart(weights, tmp_path / "bad.png", "Invalid")
