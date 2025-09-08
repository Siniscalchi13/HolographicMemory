from services.math_core import ThresholdCalculator


def test_tau_star_binary_search():
    th = ThresholdCalculator()
    # constant c_v4 (worst case) equal to 200
    c_v4_curve = lambda s: 200
    tau = th.tau_star(c_micro=200, c_v4_curve=c_v4_curve, lo=1, hi=1024)
    assert isinstance(tau, int)


def test_choose_format():
    th = ThresholdCalculator()
    fmt = th.choose_format(size=100, c_micro=16, c_v4=240, c_microk8=40)
    assert fmt in ("micro", "microK8", "v4")
    assert fmt in ("micro", "microK8")

