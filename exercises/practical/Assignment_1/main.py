import numpy as np
import pandas as pd


class DiagnosticProfile:
    def __init__(self, df: pd.DataFrame, diagnosis: [], findings: []):

        # (<attribute value>, <frequency of attribute in conjunction with a diagnose)
        self.diagnostic_profiles = {}
        for dia in diagnosis:
            self.diagnostic_profiles[dia] = {}
            for finding in findings:
                self.diagnostic_profiles[dia][finding] = {}

        for row in df.values:
            dia = row[0]
            for i in range(1, len(row)):
                if row[i] in self.diagnostic_profiles[dia][findings[i-1]].keys():
                    self.diagnostic_profiles[dia][findings[i-1]][row[i]] += 1
                else:
                    self.diagnostic_profiles[dia][findings[i - 1]][row[i]] = 1

        print(self.diagnostic_profiles)




    def prune(self, int_val):
        # do smth
        self.diagnostic_profile = []  # to be modified in some way


def exercise():
    dafr = pd.DataFrame([['cancer', 2, 1, 0],
          ['cancer', 3, 1, 0],
          ['cancer', 1, 3, 0],
          ['cancer', 1, 1, 1],
          ['no cancer', 12, 2, 3],
          ['no cancer', 2, 3, 4],
          ['no cancer', 2, 5, 6]])
    DiagnosticProfile(dafr, ['cancer', 'no cancer'], ['age', 'menopause', 'tumor-size'])


if __name__ == '__main__':
    exercise()
