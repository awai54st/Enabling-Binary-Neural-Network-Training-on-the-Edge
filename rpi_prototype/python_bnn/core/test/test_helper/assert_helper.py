import numpy as np


def assert_by_percentage(test_result_1, test_result_2, percentage=0.02, thr=1e-3):
    mask = np.abs((test_result_1-test_result_2)/(test_result_1+1e-16)) > thr
    #print(mask.sum()/test_result_1.size)
    return mask.sum()/test_result_1.size < percentage
