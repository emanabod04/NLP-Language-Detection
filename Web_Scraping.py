#%%
from encodings.utf_8 import encode
from numpy import take
import string
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import urllib.request
#%%
def read_book(book_url):
    with urllib.request.urlopen(book_url) as book_file:
        # read the entire text book
        book_file_text = book_file.read().decode('utf-8')
    return (book_file_text)
# %%
# Set the Base URL you want to scrape from
base_url='https://www.gutenberg.org'

# Connect to the URL and get html content that have all the avialable languages in gutenberg website
response = requests.get(base_url+'/browse/languages/en')


# Parse HTML and save to BeautifulSoup object
soup = BeautifulSoup(response.text, "html.parser")
paragraphs = soup.find('div', attrs={'class':'page_content'}).find_all('p')
langs=[]
for index in range(2,4):
    links = paragraphs[index].find_all('a')
    for x in links:
        num_of_books = int(re.findall(r'(\d+)',x.get('title'))[0])
        lang_name = x.get_text()
        if  num_of_books> 5 and lang_name != 'English':
            langs.append((x.get('href'),x.get_text(),num_of_books))
print(f"Number of languages: {len(langs)}")
print(langs)
# %%
lang_books_dict ={}
for lang in langs:
        response = requests.get(base_url+lang[0])
        soup = BeautifulSoup(response.text, "html.parser")
        lis = soup.find_all('li',attrs={'class':'pgdbetext'})
        list = []
        print('\n'+lang[1], end='')
        for li in lis:
            response2 = requests.get(base_url+li.a.get('href'))
            response2.encoding = 'utf-8'
            soup2 = BeautifulSoup(response2.text, "html.parser").find('a',attrs={'type':'text/plain'})
            if soup2 == None :
                soup2 = BeautifulSoup(response2.text, "html.parser").find('a',attrs={'type':'text/plain; charset=utf-8'})
            if soup2 != None :    
                link_to_book = base_url + soup2.get('href')
                book_text = read_book(link_to_book)
                list.append(book_text)
                print(' .',end='')
                if len(list) == 10:
                    break
        lang_books_dict[lang[1]] = list
# %%
print(lang_books_dict)
# %%
def clean_text(text):
       # removing the numbers
       text = re.sub(r'\d+\s|\s\d+\s|\s\d+$', ' ', text)
       # converting the text to lower case
       text = text.lower()
       #Replace punctuation with whitespaces
       text = re.compile('[%s]' % re.escape(string.punctuation)).sub('', text)
       return text
# %%
lang_part_list = []
start_match = lambda s: re.search(r'^\*\*\* START OF THIS PROJECT',s) != None
end_match = lambda s: re.search(r'^\*\*\* END OF THIS PROJECT',s) != None
def get_body_partioned(key,book_text):

    # Split the text to list of headers and paragraphs 
    book_list = re.split('\r\n[\r\n]+' ,book_text)
    start_index=0
    end_index=0
    for i,s in enumerate(book_list):
        if start_match(s)==True:
             start_index =i+1
        if end_match(s)==True:
             end_index =i-1
    new_book_text = ' '.join(book_list[start_index:end_index])
    new_book_text = clean_text(new_book_text)
    text_as_words = new_book_text.split()
    list_of_partitions = [' '.join(text_as_words[i:i+200]) for i in range(0, len(text_as_words), 200)]
    print(len(list_of_partitions))
    for part in list_of_partitions:
        lang_part_list.append((part,key))
   
    
# %%

for key in lang_books_dict:
    for book in lang_books_dict[key]:
        get_body_partioned(key,book)
    
# %%
df = pd.DataFrame.from_records(lang_part_list, columns =['Text', 'Language'])
df.to_csv('lang.csv')
# %%
print(df['Language'].value_counts())
# %%
