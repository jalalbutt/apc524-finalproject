import perturb


def test_perturbed_network():
    """
    Test PerturbedNetwork class with known test case.
    """
    m = 100  # number of grid-points
    n = 100
    L_m = 10
    L_n = 10

    f_type = "delta"
    source_center = [10, 10]
    radius = m / L_m * 0.1

    pertNet2 = perturb.PerturbedNetwork(
        f_type=f_type,
        source_center=source_center,
        m=m,
        n=n,
        L_m=L_m,
        L_n=L_n,
        radius=radius,
    )
    U = pertNet2.static_solve()
    assert abs(U[1, 10, 10] - -431.526) <= 0.01
