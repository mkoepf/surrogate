# -*- coding: utf-8 -*-
from surrogate.datahandler import load_data_file
from pandas import DataFrame


def test_load_data_file():
    data: DataFrame = load_data_file('tests/testdata.csv')

    assert(data['age'][0] == 30)
    assert(data['year'][1] == 62)
    assert(data['nodes'][2] == 0)
    assert(data['survival'][3] == 1)
