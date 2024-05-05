from bs4 import BeautifulSoup
import re

def process_text(text, section, test_df):
    """
    Input : 
    text (str) : The input text in regex format to be cleaned
    section (str) : The label of the section which is to be cleaned
    test_df (pd.DataFrame()) : The dataframe consisting of the start and end indices of all the items
    
    Output : 
    clean_text (str) : The cleaned up and processed text
    """
    
    location = 'item' + section    ## get the index value to find
    index_list = test_df.index.tolist()   ## convert index to list to find value
    curr_index = index_list.index(location)   ## get index number
    item_raw = text[test_df.iloc[curr_index]['start_idx'] : test_df.iloc[curr_index+1]['start_idx']]  ## extract the text from the locations

    #### Using BeautifulSoup remove all the tables present in the text
    soup = BeautifulSoup(item_raw, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()

    #### Extract text from the soup object removing the xml whitespace characters and tags
    text_act = soup.get_text(separator=' ', strip=True)
    text_act = text_act.replace('\xa0', ' ')

    clean_text = ' '.join(text_act.split())
    clean_text = re.sub(r'<span style=.*$', '', clean_text)
    return clean_text