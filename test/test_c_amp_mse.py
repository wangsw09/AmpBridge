import sys
sys.path.insert(0, "/home/wangsw09/work/Proj-AmpBridge/")

from AmpBridge import wrapper

TOL = 1e-7
def test_amp_mse_Lq():
    tol = 1e-9
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 1) - 0.20072727410105276) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 1.1, tol) - 0.1988871702) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 1.4, tol) - 0.1933585660) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 1.5) - 0.194643622374824) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 1.6, tol) - 0.1979671744) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 1.9, tol) - 0.2187358052) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 2) - 0.2285714285714286) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 2.1, tol) - 0.2395230168) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 2.9, tol) - 0.3497210456) < TOL
    assert abs(wrapper.mse_Lq(1, 3, 2, 0.2, 4.0, tol) - 0.5138177753) < TOL

def test_amp_mse_Lq_dalpha():
    tol = 1e-9
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 1) + 0.003410135216588) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 1.1, tol) + 0.0004263065) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 1.4, tol) + 0.0049983237) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 1.5) + 0.00937996798804) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 1.6, tol) + 0.0141753036) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 1.9, tol) + 0.0283831325) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 2) + 0.032653061224) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 2.1, tol) + 0.0366141033) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 2.9, tol) + 0.0579590031) < TOL
    assert abs(wrapper.mse_Lq_dalpha(3, 1, 2, 0.2, 4.0, tol) + 0.0682916797) < TOL

def test_amp_tau_of_alpha():
    tol = 1e-9
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 1, tol) - 2.0618111747317) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 1.1, tol) - 2.06126542305) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 1.4, tol) - 2.05979374525) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 1.5, tol) - 2.06032386434) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 1.6, tol) - 2.06151442629) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 1.9, tol) - 2.06864155408) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 2.0, tol) - 2.0720023449) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 2.1, tol) - 2.07575578482) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 2.9, tol) - 2.11466606311) < TOL
    assert abs(wrapper.tau_of_alpha(3, 1, 0.2, 0.8, 2.0, 4.0, tol) - 2.1772734091) < TOL

def test_amp_optimal_alpha():
    tol = 1e-11
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 1, tol) - 4.1607070617974) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 1.1, tol) - 3.10944362525) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 1.4, tol) - 3.55112701078) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 1.5, tol) - 4.06355141695) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 1.6, tol) - 4.7669989594) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 1.9, tol) - 8.52218739836) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 2.0, tol) - 10.5968388268) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 2.1, tol) - 13.2950014532) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 2.9, tol) - 100.617867224) < TOL
    assert abs(wrapper.optimal_alpha(1, 0.2, 0.8, 2.0, 4.0, tol) - 2225.41877958) < TOL

