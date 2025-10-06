import re
import json
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.util import ngrams  # Import ngrams

# Download necessary NLTK data (only needs to be done once)
nltk.download('stopwords')
nltk.download('punkt')

input_file = '24_ed.json'
output_file = 'cleaned_24_19_ed_data.json'  # Not used, but good practice.

# --- Define stemmer globally ---
stemmer = PorterStemmer()

# --- 1. TEXT CLEANING (Simplified)---
def clean_text(text):
    """Cleans the input text for better keyword matching."""
    #text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces to single spaces
    #text = text.translate(str.maketrans('', '', string.punctuation)) # Removes punctuation
    text = text.lower()  # Convert to lowercase
    return text.strip() #removes whitespace

# --- Helper Functions ---  # Put this *before* analyze_decision
def stem_text(text):
    """Stems the words in a given text using the Porter Stemmer."""
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# --- 2. KEYWORD ANALYSIS ---
def analyze_decision(text):
    """Analyzes a single decision text and determines the winner."""

    # --- 2.1 Extract Key Sections (using regex) ---
    conclusions_match = re.search(r"(?:conclusion(?:s)?\s+of\s+law)(.+?)(?:order(?:s)?|relief|remedies|viii?|ix|\bbased upon\b)", text, re.DOTALL | re.IGNORECASE | re.VERBOSE)
    orders_match = re.search(r"(?:order(?:s)?|relief|remedies)(.+)$", text, re.DOTALL | re.IGNORECASE | re.VERBOSE)

    conclusions = conclusions_match.group(1).strip() if conclusions_match else ""  # Group 1, not 2
    orders = orders_match.group(1).strip() if orders_match else ""

    # --- 2.2 Define Keyword Lists (Stemmed) ---
    # ---Stem the keywords

    petitioner_win_keywords = [
        "petitioner met", "petitioner prevail",
        "respondent failed",
        "relief requested.+granted",  # Use .+ to allow for other words in the phrase
        "relief is awarded",
        "conduct was a manifestation",
        "prevails on the appeal",
        "appeal.+granted",
        "GRANTED", "order to reimburse", "ORDERED to reimburse",
        "relief granted", "granted in part", "ORDERED to",
        "shall reimburse", "reimbursement.+(granted|award)",  #Reimbursement specifically
        "compensatory.+awarded", "compensatory service",
        "procedural violation.+denied fape", #looking for denial of fape language
        "did not comply",#district/respondent did NOT comply
        "hearing officer finds.+district.+(proposed.+)?(iep|program).+(not appropriate|inappropriate)", #check the endrew f cases
        "return to.+prior educational placement", #important for discipline related to MDR and FAPE
        "procedural violation.+(denied|denial).+fape", # procedural error leads to FAPE
        "procedural error", #procedural error
        "procedural right", #procedural right
        "relief is awarded",  
        "relief and order",
        
        "revise.+iep",  # Order to revise IEP
        "offer esy", #district forced to offer extended school year
        "conduct.+fba", #must perform a functional behavioral assessment.
        "draft.+ bi" #must create a behavior intervention plan.
    ]

    respondent_win_keywords = [
       "petitioner failed", 
       "petitioner did not prevail",
        "petitioner did not meet", 
        "petitioner did not show", 
        "petitioner did not demonstrate", 
        "petitioner did not establish", 
        "petitioner did not present", 
        "petitioner did not submit",
        "petitioner did not request", #petitioner errors.
        "respondent complied", 
        "respondent did not violate", 
        "district did not violate","district did not deny fape","evidence failed",
        "relief.+denied", #use .+ to cover other words
        "dismissed with prejudice",  # Very strong indicator
        "conduct is not considered a manifestation",
        "DENIED",
        "IS DISMISSED",
        "case is dismissed",
       "appropriateness of the evaluation is GRANTED",
        "request.+denied",
        "did not fail", #petitioner did not fail,
        "no violation",
        "hearing officer finds.+district.+(proposed.+)?(iep|program).+appropriate", #check the endrew f cases
        "not entitled to reimbursement", "not entitled to an iee", "not entitled to private placement",
        "provided.+fape", #district DID meet requirements.
        "reasonably calculated", #was the IEP reasonable?
        "no evidence was presented",
        "not meet state standards",
        "appropriate"
    ]

    # Stem the keywords
    petitioner_win_keywords = [stemmer.stem(keyword) for keyword in petitioner_win_keywords]
    respondent_win_keywords = [stemmer.stem(keyword) for keyword in respondent_win_keywords]

    # --- 2.3 Stem and Lowercase the Text Sections ---
    conclusions = stem_text(conclusions.lower())
    orders = stem_text(orders.lower())

    # --- 2.4 Scoring with Negation and Weights ---
    petitioner_score = 0
    respondent_score = 0

    # Weights (adjust these based on your analysis of the decisions)
    high_weight = 3
    medium_weight = 2
    low_weight = 1

    # Score "Conclusions of Law"
    for keyword in petitioner_win_keywords:
        negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b"
        if re.search(rf"\b{keyword}\b", conclusions, re.IGNORECASE):
            if re.search(negated_keyword, conclusions, re.IGNORECASE):
                respondent_score += medium_weight  # Negation of positive = negative for petitioner
            else:
                # Assign weights based on keyword (example)
                if keyword in [stemmer.stem("relief requested.+granted"), stemmer.stem("granted in part")]:
                    petitioner_score += high_weight
                elif keyword in [stemmer.stem("relief is awarded"), stemmer.stem("ORDERED to reimburse")]:
                     petitioner_score += medium_weight
                else:
                    petitioner_score += low_weight

    for keyword in respondent_win_keywords:
        negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b"  # Escape!
        if re.search(rf"\b{keyword}\b", conclusions, re.IGNORECASE):
            if re.search(negated_keyword, conclusions, re.IGNORECASE):
                petitioner_score += medium_weight  #negation
            else:
                if keyword in [stemmer.stem("relief.+denied"), stemmer.stem("dismissed with prejudice")]:
                    respondent_score += high_weight
                elif keyword in [stemmer.stem("respondent complied")]:
                    respondent_score += medium_weight

                else:
                    respondent_score += low_weight

    # Score "Orders" (Higher Weights)
    for keyword in petitioner_win_keywords:
        negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b"  # Escape!
        if re.search(rf"\b{keyword}\b", orders, re.IGNORECASE):
            if re.search(negated_keyword, orders, re.IGNORECASE):
                respondent_score += medium_weight * 1.5  # Increased weight for "Orders"
            else:
                if keyword in [stemmer.stem("relief requested.+granted"), stemmer.stem("granted in part")]:
                    petitioner_score += high_weight * 1.5
                elif keyword in [stemmer.stem("relief is awarded"), stemmer.stem("ORDERED to reimburse")]:
                    petitioner_score += medium_weight* 1.5
                else:
                    petitioner_score += low_weight * 1.5

    for keyword in respondent_win_keywords:
      negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b"  # Escape!
      if re.search(rf"\b{keyword}\b", orders, re.IGNORECASE):
          if re.search(negated_keyword, orders, re.IGNORECASE):
              petitioner_score += medium_weight * 1.5
          else:
              if keyword in [stemmer.stem("relief.+denied"), stemmer.stem("dismissed with prejudice")]:
                  respondent_score += high_weight * 1.5

              elif keyword in [stemmer.stem("respondent complied")]:
                  respondent_score += medium_weight * 1.5
              else:
                  respondent_score += low_weight * 1.5

    # --- 2.5. Determine the Winner (Adjust Threshold) ---
    if petitioner_score > respondent_score + 1.0:  # Increased threshold
        winner = "Petitioner"
    elif respondent_score > petitioner_score + 1.0:
        winner = "Respondent"
    elif petitioner_score > 0 and respondent_score > 0:  # Both have some points
        winner = "Mixed"
    else:
        winner = "Unknown"  # Truly ambiguous

    return {
        'winner': winner,
        'petitioner_score': petitioner_score,
        'respondent_score': respondent_score,
        'conclusions': conclusions,  # Return stemmed, lowercased text
        'orders': orders,            # Return stemmed, lowercased text
    }

    # --- Helper Functions ---

# --- 3. MAIN PROCESSING LOOP ---
results = []

with open(input_file, 'r', encoding='utf-8') as f:
   data = json.load(f)

for case in data:
   try:
       docket = case['docket']
       text = case['text']
       # No need to clean here - analyze_decision handles it
       analysis = analyze_decision(text) #pass the full text

       results.append({
           'docket': docket,
           'winner': analysis['winner'],
           'petitioner_score': analysis['petitioner_score'],
           'respondent_score': analysis['respondent_score'],
           'conclusions': analysis['conclusions'], # Keep for review
           'orders': analysis['orders'],           # Keep for review
       })

   except KeyError as e:
       print(f"Skipping entry due to missing key: ")
   except Exception as e:
       print(f"An unexpected error occurred: ")
       # For debugging, print the specific exception:
import traceback
traceback.print_exc()

# --- 4. CREATE DATAFRAME and DISPLAY/SAVE ---
df = pd.DataFrame(results)

# Save to CSV *before* any filtering, for complete record-keeping
df.to_csv('decision_analysis.csv', index=False)

# Print the DataFrame
# Get counts of each outcome
print("\n--- Outcome Counts ---")
print(df['winner'].value_counts())