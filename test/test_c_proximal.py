import sys
sys.path.insert(0, "/home/wangsw09/work/Proj-AmpBridge/")

from AmpBridge import wrapper

TOL = 1e-7
def test_prox_proxLq():
    tol = 1e-9
    assert wrapper.proxLq(10, 8.5, 1) == 1.5
    assert abs(wrapper.proxLq(10, 8.5, 1.1, tol) - 0.8264916771) < TOL
    assert abs(wrapper.proxLq(-10, 8.5, 1.4, tol) + 0.5604207296) < TOL
    assert abs(wrapper.proxLq(10, 8.5, 1.5) - 0.5494110372494438) < TOL
    assert abs(wrapper.proxLq(-10, 8.5, 1.6, tol) + 0.5455435539) < TOL
    assert abs(wrapper.proxLq(10, 8.5, 1.9, tol) - 0.5512342249) < TOL
    assert abs(wrapper.proxLq(-10, 8.5, 2) + 0.55555555555555556) < TOL
    assert abs(wrapper.proxLq(10, 8.5, 2.1, tol) - 0.5603637530) < TOL
    assert abs(wrapper.proxLq(-10, 8.5, 2.9, tol) + 0.6019895433) < TOL
    assert abs(wrapper.proxLq(10, 8.5, 4.0, tol) - 0.6502890228) < TOL

def test_prox_proxLq_dx():
    tol = 1e-9
    assert wrapper.proxLq_dx(10, 8.5, 1) == 1.0
    assert abs(wrapper.proxLq_dx(-10, 8.5, 1.1, tol) - 0.4739485777) < TOL
    assert abs(wrapper.proxLq_dx(10, 8.5, 1.4, tol) - 0.1292407990) < TOL
    assert abs(wrapper.proxLq_dx(-10, 8.5, 1.5) - 0.1041595659339969) < TOL
    assert abs(wrapper.proxLq_dx(10, 8.5, 1.6, tol) - 0.08773311014) < TOL
    assert abs(wrapper.proxLq_dx(-10, 8.5, 1.9, tol) - 0.06087539609) < TOL
    assert abs(wrapper.proxLq_dx(10, 8.5, 2) - 0.055555555555555556) < TOL
    assert abs(wrapper.proxLq_dx(-10, 8.5, 2.1, tol) - 0.05120299849) < TOL
    assert abs(wrapper.proxLq_dx(10, 8.5, 2.9, tol) - 0.03261364794) < TOL
    assert abs(wrapper.proxLq_dx(-10, 8.5, 4.0, tol) - 0.02265861047) < TOL

def test_prox_proxLq_dt():
    tol = 1e-9
    assert wrapper.proxLq_dt(10, 8.5, 1) == -1
    assert abs(wrapper.proxLq_dt(10, 8.5, 1.1, tol) + 0.5115024967) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 1.4, tol) + 0.1435269138) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 1.5) + 0.115808146374) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 1.6, tol) + 0.0975845728) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 1.9, tol) + 0.0676702775) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 2) + 0.061728395062) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 2.1, tol) + 0.0568632565) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 2.9, tol) + 0.0360592240) < TOL
    assert abs(wrapper.proxLq_dt(10, 8.5, 4.0, tol) + 0.0249237011) < TOL

def test_prox_proxLq_inv():
    assert abs(wrapper.proxLq_inv(10, 8.5, 1.1) - 21.770952600275) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 1.4) - 39.891448534964) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 1.5) - 50.319040167147) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 1.6) - 64.142575195276) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 1.9) - 138.284009907971) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 2) - 180) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 2.1) - 234.718186005259) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 2.9) - 1968.019098595354) < TOL
    assert abs(wrapper.proxLq_inv(10, 8.5, 4.0) - 34010) < TOL

