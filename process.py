from bs4 import BeautifulSoup
import re

def process_text(text, section, test_df):
    location = 'item' + section
    index_list = test_df.index.tolist()
    curr_index = index_list.index(location)
    item_raw = text[test_df.iloc[curr_index]['start_idx'] : test_df.iloc[curr_index+1]['start_idx']]
    soup = BeautifulSoup(item_raw, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()

    text_act = soup.get_text(separator=' ', strip=True)
    text_act = text_act.replace('\xa0', ' ')

    clean_text = ' '.join(text_act.split())
    clean_text = re.sub(r'<span style=.*$', '', clean_text)
    return clean_text