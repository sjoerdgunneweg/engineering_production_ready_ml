from src.telemetry.drift_calculation import get_psi


def test_get_psi():
    training = (0.5, 0.5)
    latest = (0.6, 0.4)

    expected = 0.04054651

    out = get_psi(training, latest)  

    assert round(out, 8) == expected


def test_get_psi_when_identical():
    training = (0.7, 0.3)
    latest = (0.7, 0.3)

    expected = 0.0

    out = get_psi(training, latest)

    assert out == expected

