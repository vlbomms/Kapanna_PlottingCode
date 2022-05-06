import sys, os
import glob
import json
import MRIReport
from datetime import datetime

# For local testng
#os.environ["WORKFLOW_DIR"] = "/data"
#os.environ["BATCH_NAME"] = "batch"
#os.environ["OPERATOR_OUT_DIR"] = "output"
#os.environ["OPERATOR_IN_DCM_METADATA_DIR"] = "dcm2json"
#os.environ["OPERATOR_IN_DIR_BMASK"] = "None"
#os.environ["OPERATOR_IN_DIR_ICV"] = "None"
#os.environ["OPERATOR_IN_DIR_ROI"] = "deepmrseg"

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:
    bmask = []
    icv = []
    roi = []

    print(f'Checking for nrrd/json files')

    if "None" not in os.environ["OPERATOR_IN_DIR_BMASK"]:
        print("bmask folder provided")
        bmask_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_BMASK'])
        bmask = sorted(glob.glob(os.path.join(bmask_input_dir, "*.nrrd*"), recursive=True))
    if "None" not in os.environ["OPERATOR_IN_DIR_ICV"]:
        print("icv folder provided")
        icv_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_ICV'])
        icv = sorted(glob.glob(os.path.join(icv_input_dir, "*.nrrd*"), recursive=True))
    if "None" not in os.environ["OPERATOR_IN_DIR_ROI"]:
        print("roi folder provided")
        roi_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_ROI'])
        roi = sorted(glob.glob(os.path.join(roi_input_dir, "*.nrrd*"), recursive=True))

    tmp_json = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DCM_METADATA_DIR'])

    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    if not os.path.exists(element_output_dir):
        os.makedirs(element_output_dir)
    
    # The processing algorithm
    json_file = sorted(glob.glob(os.path.join(tmp_json, "*.json*"), recursive=True))
    
    if len(bmask) == 0 and len(icv) == 0 and len(roi) == 0:
        print("No nrrd file(s) found!")
        exit(1)
    elif len(json_file) != 1:
        print("Incorrect # of JSON file in directory")
        exit(1)
    else:
        json_file = json_file[0]
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)

        pdf_file_path = os.path.join(element_output_dir, "{}.pdf".format(os.path.basename(batch_element_dir)))
        print("Executing pdf creation")
        MRIReport._main(bmask, roi, icv, json_file, pdf_file_path)