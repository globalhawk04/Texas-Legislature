import requests
import PyPDF2
import io
import json
import time
import urllib
import urllib.request
from bs4 import BeautifulSoup
import os
import re
import string
import json
import time
import pandas as pd
import csv

create_json = []

reference_list = []
final_list = []
hearings = [
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2024',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2023',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2022',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2021',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2020',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2019',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2018',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2017',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2016',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2015',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2014',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2013',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2012',
#'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/special-education-due-process-hearings-2011',
'https://tea.texas.gov/about-tea/government-relations-and-legal/special-education-hearings/due-process-hearings/due-process-hearing-decisions-issued-in-2010',
]


all_list = []
def make_soup(url):
    thepage = urllib.request.urlopen(url)
    soupdata = BeautifulSoup(thepage, "lxml")
    return soupdata
all_documents = []
for hearings_by_year in hearings:
    soup = make_soup(hearings_by_year)

    all_info = []
    for table_info in soup.findAll('tr'):
        #print(table_info.text)
        x = table_info.text.split(' ')
        print(x)
        together = []
        for y in x:
            #print(len(y))
            if len(y) == 0:
                pass
            else:
                together.append(y)
                #print(y)

        for href in table_info.findAll('a'):
            #print(href.get('href')) 
     
            x.append('https://tea.texas.gov'+href.get('href'))
            def extract_text_from_pdf(url):
                try:
                    # Download the PDF
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    # Create a file-like object from the response content
                    pdf_file = io.BytesIO(response.content)
                    # Create a PDF reader object
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    # Extract text from each page
                    text = ""
                    pages = []
                    for page_num in range(len(pdf_reader.pages)):
                        pages.append(page_num)
                        #page = pdf_reader.pages[page_num]
                        #text += page.extract_text()
                    for last_pages in pages[-2:]:
                        page = pdf_reader.pages[last_pages]
                        text += page.extract_text()
                        return text
                except requests.exceptions.RequestException as e:
                    print(f"Error during request: {e}")
                    return None
                except PyPDF2.errors.PdfReadError as e:
                    print(f"Error reading PDF: {e}")
                    return None
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    return None
            if __name__ == "__main__":
                pdf_url = 'https://tea.texas.gov'+href.get('href')
                extracted_text = extract_text_from_pdf(pdf_url)
                time.sleep(1)


                try:
                    if extracted_text:
                        print(extracted_text)
                        create_json.append(dict(
                        {'docket': x[0],
                         'petition': x[1],
                         'respondent':x[2],
                         'date':x[4],
                         'hearing_officer':x[5],
                         #'key_words':x[5],
                         'text':extracted_text,
                        }))
                except Exception as e:
                    print(e)
                    pass

        try:
            json_data = json.dumps(create_json, indent=4)
            with open('2010_Due_Process.json','w') as f:
                f.write(json_data)
        except Exception as e:
            print(e)
