# -*- coding: utf-8 -*-
from typing import List, Dict

from surrogate.datahandler import load_data_file

def test_load_data_file():
    data: List[Dict] = load_data_file('tests/testdata.csv')

    assert(data[0]['age'] == 30)
    assert(data[1]['year'] == 62)
    assert(data[2]['nodes'] == 0)
    assert(data[3]['survival'] == 1)
