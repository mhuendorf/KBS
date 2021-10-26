import numpy as np
import pandas as pd


class DiagnosticProfile:
    def __init__(self, df: pd.DataFrame, diagnosis: [], findings: []):

        # (<attribute value>, <frequency of attribute in conjunction with a diagnose)
        self.diagnostic_profiles = {}

        # initialize diagnostic_profiles structure
        for dia in diagnosis:
            self.diagnostic_profiles[dia] = {}
            for finding in findings:
                self.diagnostic_profiles[dia][finding] = {}

        # number of cases of every diagnosis
        self.num_cases = {}
        # zero cases at the start
        for diagnose in diagnosis:
            self.num_cases[diagnose] = 0

        for row in df.values:
            dia = row[0]
            self.num_cases[dia] += 1
            for i in range(1, len(row)):
                if row[i] in self.diagnostic_profiles[dia][findings[i-1]].keys():
                    self.diagnostic_profiles[dia][findings[i-1]][row[i]] += 1
                else:
                    self.diagnostic_profiles[dia][findings[i - 1]][row[i]] = 1
        # normalize values to match frequencies
        for dia in self.diagnostic_profiles.keys():
            cases = self.num_cases[dia]
            for attr in self.diagnostic_profiles[dia]:
                for attr_val in self.diagnostic_profiles[dia][attr].keys():
                    self.diagnostic_profiles[dia][attr][attr_val] /= cases

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
