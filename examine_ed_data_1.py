import re
import json

import nltk
import pandas as pd 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
nltk.download('stopwords')
nltk.download('punkt_tab')

input_file = '24_19_ed_data.json'
output_file = 'cleaned_24_19_ed_data.json'

def clean_text(text):
    text = re.sub(r'\s+', ' ',text)
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b',' ')
    text = text.replace('b/n/f',' ')
    text = text.replace('/s/',' ')
    text = text.replace('  ',' ')
    text = text.replace('__',' ')
    text = text.replace('\t',' ')
    text = text.replace('\r',' ')

    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))\
    #tokenize text
    word_tokens = word_tokenize(text)
    # remove stop words
    #stop_words = set(stopwords.words('english'))
    #filtered_words = [word for word in word_tokens if word.lower()not in stop_words]
    #join back into a string
    #text = ' '.join(filtered_words)
    return text.strip()

def analyze_decision(text):
    """Analyzes a single decision text and determines the winner."""

    # --- 1. Extract Key Sections (using regex) ---
    conclusions_match = re.search(r"(VII?|VIII?|IX)\s+CONCLUSIONS\s+OF\s+LAW(.+?)(VIII?|IX|\bORDERS?\b|\bORDER\b|\bRELIEF\b)", text, re.DOTALL | re.IGNORECASE)
    orders_match = re.search(r"(VIII?|IX|\bORDERS?\b|\bORDER\b|\bRELIEF\b)(.+)$", text, re.DOTALL | re.IGNORECASE)

    conclusions = conclusions_match.group(2).strip() if conclusions_match else ""
    orders = orders_match.group(2).strip() if orders_match else ""

    # --- 2. Define Keyword Lists (for Petitioner and Respondent wins) ---

    petitioner_win_keywords = [
        "petitioner met",
        "respondent failed",
        "relief requested.+granted",  # Use .+ to allow for other words
        "relief is awarded",
        "conduct was a manifestation",
      	"prevails on the appeal",
        "GRANTED",
        "ORDERED to reimburse"
    ]

    respondent_win_keywords = [
        "petitioner failed",
        "petitioner did not meet",
        "respondent complied",
        "relief.+denied", #use .+ to cover other words
        "dismissed with prejudice",  # Very strong indicator
        "conduct is not considered a manifestation",
        "DENIED",
        "IS DISMISSED",
        "appropriateness of the evaluation is GRANTED"
    ]
    # ---Lowercase all keywords
    petitioner_win_keywords = [keyword.lower() for keyword in petitioner_win_keywords]
    respondent_win_keywords = [keyword.lower() for keyword in respondent_win_keywords]
    # --- 3. Score the Decision ---
    #convert conclusions and orders strings to lowercase
    conclusions = conclusions.lower()
    orders = orders.lower()

    petitioner_score = 0
    respondent_score = 0

    # Score "Conclusions of Law"
    for keyword in petitioner_win_keywords:
        if re.search(rf"\b{keyword}\b", conclusions):  # Use word boundaries (\b)
            petitioner_score += 1
    for keyword in respondent_win_keywords:
        if re.search(rf"\b{keyword}\b", conclusions):
            respondent_score += 1

    # Score "Orders" (give slightly more weight)
    for keyword in petitioner_win_keywords:
        if re.search(rf"\b{keyword}\b", orders):
            petitioner_score += 1.5  # Slightly higher weight
    for keyword in respondent_win_keywords:
        if re.search(rf"\b{keyword}\b", orders):
            respondent_score += 1.5

    # --- 4. Determine the Winner ---
    if petitioner_score > respondent_score + 0.5: #added the padding again
        winner = "Petitioner"
    elif respondent_score > petitioner_score + 0.5:
        winner = "Respondent"
    elif petitioner_score > 0 and respondent_score > 0:
        winner = "Mixed"
    else: #if its == then mixed otherwise unknown for a 0,0 pt score
        winner = "Unknown" #added the else condition

    return {
        'winner': winner,
        'petitioner_score': petitioner_score,
        'respondent_score': respondent_score,
        'conclusions': conclusions,  # Include for analysis
        'orders': orders,          # Include for analysis
    }
original_data = []
count_bill_sum_text = []
count_bill_enr_text = []
percent = []
cleanded_data = []
skipped_count = 0

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


total_entries = len(data)
for key in data:
    #print(key)
    original_data.append(key)
    try:
        docket = key['docket']
        text = key['text']
        petition = key['petition']
        respondent = key['respondent']
        date = key['date']
        cleaned_text = clean_text(text)
        winner = analyze_decision(cleaned_text)
        print(winner['winner'])

    except Exception as e:
        print(e)
