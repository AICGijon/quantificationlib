from quantificationlib.metrics.multiclass import hd, bray_curtis, jensenshannon, kld, geometric_mean
from quantificationlib.metrics.binary import absolute_error, bias, relative_absolute_error, squared_error, symmetric_absolute_percentage_error, normalized_absolute_score, normalized_squared_score
from numpy.testing import assert_almost_equal

def test_metrics_quant_multiclass():
    p_true = [0.1, 0.4, 0.5]
    p_hat = [0.15,0.35,0.5]


    assert_almost_equal(hd(p_true, p_hat), 0.08197, decimal=5, err_msg="error in hd")
    assert_almost_equal(bray_curtis(p_true, p_hat), 0.05, decimal=5, err_msg="error in bray_curtis")
    assert_almost_equal(jensenshannon(p_true, p_hat), 0.00335, decimal=5, err_msg="error in jensenshannon")
    assert_almost_equal(kld(p_true, p_hat), 0.01856, decimal=5, err_msg="error in kld")

def test_metrics_quant_binary():
    p_true = 0.7
    p_hat = 0.6

    assert_almost_equal(bias(p_true, p_hat), -0.1, decimal=5, err_msg="error in bias")
    assert_almost_equal(absolute_error(p_true, p_hat), 0.1, decimal=5, err_msg="error in absolute_error")
    assert_almost_equal(relative_absolute_error(p_true, p_hat), 0.142857, decimal=5, err_msg="error in relative absolute_error")
    assert_almost_equal(squared_error(p_true, p_hat), 0.01, decimal=5, err_msg="error in squared error")
    assert_almost_equal(symmetric_absolute_percentage_error(p_true, p_hat), 0.076923, decimal=5, err_msg="error in symmetric_absolute_percentage_error")
    assert_almost_equal(normalized_absolute_score(p_true, p_hat), 0.85714, decimal=5, err_msg="error in normalized absolute score")
    assert_almost_equal(normalized_squared_score(p_true, p_hat), 0.97959, decimal=5, err_msg="error in normalized_squared_score")


def test_geometric_mean():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 1]

    assert_almost_equal(geometric_mean(y_true, y_pred), 0.612372, decimal=5, err_msg="error in geometric_mean")


