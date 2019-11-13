# -*- coding: utf-8 -*-
"""
Usage:
  # Convert xml to csv
  python xml_to_csv.py --input_path=/path/to/xml/dir --output=xxx.csv
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='dir path to xmls')
parser.add_argument('--output', type=str, help='save name')
args = parser.parse_args()


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if '0-1572245415' in root.find('filename'):
            print(path)
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':
    xml_path = args.input_path
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv(args.output, index=False)
    print('Successfully converted xml to csv')
