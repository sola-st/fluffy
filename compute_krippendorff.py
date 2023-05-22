#!/usr/bin/env python
import krippendorff
import numpy as np
import csv

def main():
    with open('survey_result.csv', mode ='r')as file:        
        reader = csv.reader(file)
        next(reader) # skip headers
        reliability_data = [[np.nan if v == "*" else v for v in coder_reponses]
                            for coder_reponses in reader]

    print(reliability_data)
    # We have 3 labels (value) for each flow:
    # "yes" means that the coder consider the flow expected;
    # "no" means that the coder consider the flow unexpected;
    # "*" means that the coder does not make a decision.
    print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=reliability_data,
                                                                          level_of_measurement="nominal",
                                                                          value_domain=['yes', 'no'],
                                                                          ))


if __name__ == '__main__':
    main()
