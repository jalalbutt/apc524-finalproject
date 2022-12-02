from unc import LabUnc


def test_add():
    assert LabUnc(10, 1) + LabUnc(10, 2) == LabUnc(20, 3)
    assert LabUnc(20, 0.1) + LabUnc(10, 0.1) == LabUnc(30, 0.2)


def test_sub():
    assert LabUnc(10, 1) - LabUnc(10, 2) == LabUnc(0, 3)
    assert LabUnc(20, 0.1) - LabUnc(10, 0.1) == LabUnc(10, 0.2)


def test_mul():
    assert LabUnc(10, 1) * LabUnc(10, 2) == LabUnc(100, 30)
    assert LabUnc(20, 0.1) * LabUnc(10, 0.1) == LabUnc(200, 3)


def test_div():
    assert LabUnc(10, 1) / LabUnc(10, 2) == LabUnc(1, 0.3)
    assert LabUnc(20, 0.1) / LabUnc(10, 0.1) == LabUnc(2, 0.03)


def test_pow():
    assert LabUnc(10, 1) ** 2 == LabUnc(100, 20)
    assert LabUnc(20, 0.1) ** 3 == LabUnc(8000, 120)
