import re
import pandas as pd

def get_section_location(text):
    """
    Input :
    text (str) : The text from which the location of sections need to be extracted
    
    Output :
    text (str) : The input text
    test_df (pd.DataFrame()) : A datframe consisting of all the start locations and ending locations of the relevant sections (items)
    """
    
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;|&#xA0;)(1\.|1A|1B|1C|2|3|4|5|6|7|7A|8)\.{0,1})|(>ITEM(\s|&#160;|&nbsp;|&#xA0;)(1\.|1A|1B|1C|2|3|4|5|6|7|7A|8))')  ### Check for the relevant patterns in the text (signifying the start and end of the item)
    matches = regex.finditer(text)   ## Get all the matches for the patterns

    ### Create a dataframe with the name, start and end index of each item
    test_df = pd.DataFrame([[x.group(), x.start(), x.end()] for x in matches])
    test_df.columns = ['group', 'start_idx', 'end_idx']
    test_df.group = test_df.group.str.lower()

    #### Replace all the regex characters with blank spaces to check for duplicates
    test_df.replace('&#160;',' ',regex=True,inplace=True)
    test_df.replace('&nbsp;',' ',regex=True,inplace=True)
    test_df.replace('&#xa0;',' ',regex=True,inplace=True)

    test_df.replace(' ','',regex=True,inplace=True)
    test_df.replace('\.','',regex=True,inplace=True)
    test_df.replace('>','',regex=True,inplace=True)

    #### Drop the duplicates and only keep the last value which contains the actual text
    test_df = test_df.drop_duplicates('group', keep = 'last')
    test_df.index = range(len(test_df))
    test_df = test_df.set_index('group')
    
    return text, test_df