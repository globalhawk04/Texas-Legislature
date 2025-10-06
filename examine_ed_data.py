#check out the ed data

# I have the file i need to examine
# clean it up then search it for legal terms
# pull those terms into a column
# manually go through to double check human intellegince
# ask the ai how it would inference easier but i think it filters through "what its learned for context" 


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

# --- 2. N-GRAM EXTRACTION ---
def extract_ngrams(text, n=2):
    """Extracts n-grams from the text, removing stop words and punctuation."""
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    # Remove stop words AND non-alphabetic tokens
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    n_grams = ngrams(filtered_tokens, n)
    return [' '.join(gram) for gram in n_grams]

# --- 3. FREQUENCY ANALYSIS ---
def frequency_analysis(ngram_list):
    """Performs frequency analysis on a list of n-grams."""
    frequency = {}
    for ngram in ngram_list:
        frequency[ngram] = frequency.get(ngram, 0) + 1
    return frequency

# --- 4. MAIN PROCESSING LOOP ---

all_ngrams = []  # Collect n-grams from ALL documents

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for case in data:
    try:
        text = case['text']

        # Extract bigrams (n=2) and trigrams (n=3)
        bigrams = extract_ngrams(text, 2)
        trigrams = extract_ngrams(text, 3)

        all_ngrams.extend(bigrams)
        all_ngrams.extend(trigrams)

        #--- You can add other n-gram sizes if you like---
        #fourgrams = extract_ngrams(text, 4)
        #all_ngrams.extend(fourgrams)


    except KeyError as e:
        print(f"Skipping entry due to missing key: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- 5. OVERALL FREQUENCY ANALYSIS ---
ngram_frequency = frequency_analysis(all_ngrams)

# --- 6. CREATE PANDAS DATAFRAME ---
frequency_df = pd.DataFrame(ngram_frequency.items(), columns=['ngram', 'count'])
frequency_df = frequency_df.sort_values(by='count', ascending=False)


# --- 7. DISPLAY AND SAVE RESULTS ---
print("Top 20 Most Frequent N-grams:")
print(frequency_df.head(20))

# Save to CSV (essential for review)
frequency_df.to_csv('ngram_frequency.csv', index=False)

# data storage

original_data = []
count_bill_sum_text = []
count_bill_enr_text = []
percent = []
cleanded_data = []
skipped_count = 0

# load data

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

total_entries = len(data)

#print(total_entries)	

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
		print(cleaned_text)
	except Exception as e:
		print(e)





