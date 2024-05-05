import re
import pandas as pd

def get_section_location(text):
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;|&#xA0;)(1\.|1A|1B|1C|2|3|4|5|6|7|7A|8)\.{0,1})|(>ITEM(\s|&#160;|&nbsp;|&#xA0;)(1\.|1A|1B|1C|2|3|4|5|6|7|7A|8))')
    matches = regex.finditer(text)

    test_df = pd.DataFrame([[x.group(), x.start(), x.end()] for x in matches])
    test_df.columns = ['group', 'start_idx', 'end_idx']
    test_df.group = test_df.group.str.lower()
    
    test_df.replace('&#160;',' ',regex=True,inplace=True)
    test_df.replace('&nbsp;',' ',regex=True,inplace=True)
    test_df.replace('&#xa0;',' ',regex=True,inplace=True)

    test_df.replace(' ','',regex=True,inplace=True)
    test_df.replace('\.','',regex=True,inplace=True)
    test_df.replace('>','',regex=True,inplace=True)

    test_df = test_df.drop_duplicates('group', keep = 'last')
    test_df.index = range(len(test_df))
    test_df = test_df.set_index('group')
    
    return text, test_df