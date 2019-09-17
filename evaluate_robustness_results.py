import numpy as np
import os
from pprint import pprint as pp

import store_load as sl

if __name__ == "__main__":
    print("evaluate robustness results")
    datatype = "1_compress_10"
    base_path = 'Analysis_data/robustness/'
    tests = os.listdir(base_path)
    # filter sameSeed
    filtered = [file for file in tests if '1_' in file]
    test_cases = filtered
    pp(test_cases)
    name = "high_contrast_"
    role = "robustness"
    f = open("Analysis_data/robustness/robustness_eval.txt", "w+")
    nl = "\n"
    header0 = "total similarity difference: negative value means that the manipulated version is less" + \
              "similar to the original" + nl
    header1 = "test case ; total similarity difference ; detect manipulated ; detect base" + nl
    f.write(header0)
    f.write(header1)
    for datatype in test_cases:
        datapaths = sl.get_datapaths_by_name(datatype, role, name)

        for data in datapaths:
            img_name = os.path.basename(data)
            base_similarity = sl.load_data(sl.get_datapaths_by_name("orig_1", role, img_name)[0])
            total_diff = abs(sl.load_data(data)[0]) - abs(base_similarity[0])
            detect_manipulated = sl.load_data(data)[1]
            detect_base = base_similarity[1]
            case = img_name
            result = f"{case} ; {total_diff} ; {detect_manipulated} ; {detect_base}" + nl
            f.write(result)
    f.close()
