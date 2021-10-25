import numpy as np
import pandas as pd


class DiagnosticProfile:
    def __init__(self, df: pd.DataFrame, diagnosis: [], findings: []):
        # list of tuples of
        # (<attribute value>, <frequency of attribute in conjunction with a diagnose)
        self.diagnostic_profile = []

    def prune(self, int_val):
        # do smth
        self.diagnostic_profile = []  # to be modified in some way


def exercise():
    print("test")


if __name__ == '__main__':
    exercise()
