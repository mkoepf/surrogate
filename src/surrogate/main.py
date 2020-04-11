# -*- coding: utf-8 -*-
"""
Trains a (simple) neuronal network and a human-interpretable decision tree at
the same time.
"""

import argparse
import sys
import logging

from pandas import DataFrame
import sklearn.linear_model as lin

from surrogate import __version__, datahandler

__author__ = "Michael Köpf"
__copyright__ = "Michael Köpf"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Train a dNN together with a surrogate model")
    parser.add_argument(
        "--version",
        action="version",
        version="surrogate {ver}".format(ver=__version__))
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Starting ...")

    patient_data: DataFrame = datahandler.load_data_file('data/haberman/haberman.data')

    # Split data into training and test set
    data_test = patient_data[:50]
    data_train = patient_data[50:]

    # Train, i.e., fit
    x_train = data_train.drop(columns=['survival'])
    y_train = data_train['survival']
    clf = lin.LogisticRegression().fit(x_train, y_train)

    print(clf.coef_)

    # Calculate accuracy on the test set
    x_test = data_test.drop(columns=['survival'])
    y_test = data_test['survival']
    score = clf.score(x_test, y_test)

    print(score)

    _logger.info("Done!")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
