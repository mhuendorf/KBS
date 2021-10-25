import numpy as np
import pandas as pd


class DiagnosticProfile:
    def __init__(self, df: pd.DataFrame, diagnosis: [], findings: []):

        # (<attribute value>, <frequency of attribute in conjunction with a diagnose)
        self.diagnostic_profiles = {}
        for dia in diagnosis:
            self.diagnostic_profiles[dia] = {}
            for finding in findings:
                self.diagnostic_profiles[dia][finding] = [('value', 'freq'), ('value', 'freq')]


    def prune(self, int_val):
        # do smth
        self.diagnostic_profile = []  # to be modified in some way


def exercise():
    DiagnosticProfile(pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1)), ['cancer', 'no cancer'], ['age', 'menopause', 'tumor-size'])


if __name__ == '__main__':
    exercise()
