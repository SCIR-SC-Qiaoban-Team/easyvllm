################################################################################
# Copyright 2025 XingYuSSS
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import os

import pandas as pd


def read_file(file_name):
    _, file_extension = os.path.splitext(file_name)
    
    if file_extension == '.csv':
        df = pd.read_csv(file_name)
    elif file_extension == '.json':
        df = pd.read_json(file_name)
    elif file_extension == '.jsonl':
        df = pd.read_json(file_name, lines=True)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    return df

def save_file(df: pd.DataFrame, file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.csv':
        df.to_csv(file_path, index=False)
        print(f"DataFrame successfully saved as CSV: {file_path}")
    
    elif file_extension == '.json':
        df.to_json(file_path, orient='records', lines=False, indent=2, force_ascii=False)
        print(f"DataFrame successfully saved as JSON: {file_path}")
    
    elif file_extension == '.jsonl':
        df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        print(f"DataFrame successfully saved as JSONL: {file_path}")
    
    elif file_extension == '.xlsx' or file_extension == '.xls':
        df.to_excel(file_path, index=False)
        print(f"DataFrame successfully saved as Excel: {file_path}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")