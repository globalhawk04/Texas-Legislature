import re
import json
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.util import ngrams  # Import ngrams (you had this, just making sure)

# Download necessary NLTK data (only needs to be done once)
nltk.download('stopwords')
nltk.download('punkt')

input_file = '24_ed.json'
#output_file = 'cleaned_24_19_ed_data.json'  # Not currently used, but good practice


# --- 1.  TEXT CLEANING (Simplified and Improved) ---
def clean_text(text):
    """Cleans the input text for better keyword matching.
       Removes whitespace, punctuation, and converts to lowercase.
    """
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces to single spaces
    text = text.replace('\xa0', ' ').replace('\u200b', ' ').replace('b/n/f', ' ').replace('/s/', ' ')
    text = text.replace(' ', ' ').replace('__', ' ').replace('\t', ' ').replace('\r', ' ')  # Remove specific unwanted characters
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text.strip()  # Remove leading/trailing whitespace



# --- 2. N-GRAM EXTRACTION (Optional, for future expansion) ---
def extract_ngrams(text, n=2):
    """Extracts n-grams from cleaned text, removing stop words and punctuation."""
    cleaned_text = clean_text(text)  # Clean the text first
    tokens = word_tokenize(cleaned_text)  # Tokenize into words
    stop_words = set(stopwords.words('english'))
    # Remove stop words and non-alphabetic tokens *before* creating n-grams
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    n_grams = ngrams(filtered_tokens, n)  # Create the n-grams
    return [' '.join(gram) for gram in n_grams]  # Join into strings


# --- 3. STEMMING HELPER FUNCTION ---
stemmer = PorterStemmer()  # Initialize the stemmer *globally* (best for this script)

def stem_text(text):
    """Stems the words in a given text using the Porter Stemmer."""
    tokens = word_tokenize(text)  # Tokenize the text
    stemmed_tokens = [stemmer.stem(token) for token in tokens]  # Stem each token
    return ' '.join(stemmed_tokens)  # Join back into a string


# --- 4. KEYWORD ANALYSIS (The Core Logic) ---

def analyze_decision(text):
    """Analyzes a single decision text and determines the winner.
    This function extracts key sections, applies keyword matching with
    stemming and negation handling, and scores the results.
    """

    # --- 4.1. Extract Key Sections ---
    # Use re.VERBOSE to allow comments and whitespace in the regex (for readability)
    # Use re.IGNORECASE to make the search case-insensitive.
    # Use re.DOTALL so that '.' matches any character, including newlines.
    # Use (?:...) for non-capturing groups (more efficient).

    conclusions_match = re.search(
        r"(?:conclusion(?:s)?\s+of\s+law)(.+?)(?:order(?:s)?|relief|remedies|viii?|ix|\bbased upon\b)",
        text, re.DOTALL | re.IGNORECASE | re.VERBOSE)

    orders_match = re.search(
        r"(?:order(?:s)?|relief|remedies)(.+)$",
        text, re.DOTALL | re.IGNORECASE | re.VERBOSE
    )
    
    # Safely get the matched text, defaulting to "" if no match is found.  Use group(1).
    conclusions = conclusions_match.group(1).strip() if conclusions_match else ""
    orders = orders_match.group(1).strip() if orders_match else ""

    # --- 4.2. Define Keyword Lists (Stemmed) ---
    # These lists contain keywords and phrases indicative of Petitioner or Respondent wins.
    # Regular expressions are used for flexibility (e.g., "relief.+granted").
    
    petitioner_win_keywords = [
        "petitioner met", "petitioner prevail",
        "respondent failed",
        "relief requested.+granted",  # Use .+ to allow for other words
        "relief is awarded",
        "conduct was a manifestation",
        "prevails on the appeal",
        "GRANTED", "order to reimburse", "ORDERED to reimburse",
        "relief granted", "granted in part", "ORDERED to",
        "shall reimburse", "reimbursement.+(granted|award)",  #Reimbursement specifically
        "compensatory.+awarded", "compensatory service",
        "procedural violation.+denied fape", #looking for denial of fape language
        "did not comply", #district/respondent did NOT comply
        "hearing officer finds.+district.+(proposed.+)?(iep|program).+(not appropriate|inappropriate)", #check the endrew f cases

    ]

    respondent_win_keywords = [
       "petitioner failed", "petitioner did not prevail",
        "petitioner did not meet",
        "respondent complied", "respondent did not violate", "district did not violate","district did not deny fape",
        "relief.+denied", #use .+ to cover other words
        "dismissed with prejudice", # Very strong indicator
        "conduct is not considered a manifestation",
        "DENIED",
        "IS DISMISSED",
        "case is dismissed",
       "appropriateness of the evaluation is GRANTED",
        "request.+denied",
        "did not fail", #petitioner did not fail,
        "no violation",
        "hearing officer finds.+district.+(proposed.+)?(iep|program).+appropriate", #check the endrew f cases
    ]


    # Stem the keywords *once* (more efficient than stemming in the loop)
    petitioner_win_keywords = [stemmer.stem(keyword) for keyword in petitioner_win_keywords]
    respondent_win_keywords = [stemmer.stem(keyword) for keyword in respondent_win_keywords]

    # --- 4.3. Stem and Lowercase the Text Sections ---
    conclusions = stem_text(conclusions.lower())
    orders = stem_text(orders.lower())

    # --- 4.4. Scoring (with Negation and Weights) ---
    petitioner_score = 0
    respondent_score = 0

    # Weights (adjust these based on your analysis of the decisions)
    high_weight = 3
    medium_weight = 2
    low_weight = 1

    # --- 4.4.1 Score "Conclusions of Law" ---
    for keyword in petitioner_win_keywords:
        # Construct a regex to check for negation *before* the keyword.
        negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b" #escape
        # re.escape is important!  It handles special regex characters.
        #If the is in the list, add points accordingly
        if re.search(rf"\b{keyword}\b", conclusions, re.IGNORECASE):  # Use word boundaries and case-insensitivity
            if re.search(negated_keyword, conclusions, re.IGNORECASE):
                respondent_score += medium_weight  # Negation of a positive = a win for the respondent
            else:
                # Assign weights based on keyword (you can customize this)
                if keyword in [stemmer.stem("relief requested.+granted"), stemmer.stem("granted in part")]:
                    petitioner_score += high_weight  # Strong indicators
                elif keyword in [stemmer.stem("relief is awarded"), stemmer.stem("ORDERED to reimburse")]:
                    petitioner_score += medium_weight #Medium indicators
                else:
                    petitioner_score += low_weight    # Default weight

    for keyword in respondent_win_keywords:
        negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b" #escape
        if re.search(rf"\b{keyword}\b", conclusions, re.IGNORECASE):
            if re.search(negated_keyword, conclusions, re.IGNORECASE):
                petitioner_score += medium_weight # Negation of negative = positive.
            else:
                if keyword in [stemmer.stem("relief.+denied"), stemmer.stem("dismissed with prejudice")]:
                    respondent_score += high_weight #High weight
                elif keyword in [stemmer.stem("respondent complied")]:
                    respondent_score += medium_weight
                else:
                    respondent_score += low_weight


    # --- 4.4.2 Score "Orders" (Same logic, but higher weights) ---
    for keyword in petitioner_win_keywords:
        negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b" #escape
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
        negated_keyword = r"\b(?:not|no|fail(?:ed)?\s+to)\s+" + re.escape(keyword) + r"\b"  #escape
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


    # --- 4.5. Determine the Winner ---
    # Use a threshold to handle cases with close scores.  Adjust as needed.
    if petitioner_score > respondent_score + 1.0:  # Increased threshold
        winner = "Petitioner"
    elif respondent_score > petitioner_score + 1.0:
        winner = "Respondent"
    elif petitioner_score > 0 and respondent_score > 0:
        winner = "Mixed"
    else:
        winner = "Unknown"

    return {
        'winner': winner,
        'petitioner_score': petitioner_score,
        'respondent_score': respondent_score,
        'conclusions': conclusions,  # Return stemmed/lowercased text for analysis
        'orders': orders,            # Return stemmed/lowercased text for analysis
    }


# --- 5. MAIN PROCESSING LOOP ---

results = []

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for case in data:
    try:
        docket = case['docket']
        text = case['text']  # Pass the *full* text to analyze_decision
        analysis = analyze_decision(text) #now the full text is being passed

        results.append({
            'docket': docket,
            'winner': analysis['winner'],
            'petitioner_score': analysis['petitioner_score'],
            'respondent_score': analysis['respondent_score'],
            'conclusions': analysis['conclusions'],  # Keep for review
            'orders': analysis['orders'],            # Keep for review
        })

    except KeyError as e:
        print(f"Skipping entry due to missing key: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # For debugging, print the specific exception:
        import traceback
        traceback.print_exc()

# --- 6. CREATE DATAFRAME and DISPLAY/SAVE ---
df = pd.DataFrame(results)

# Save to CSV *before* any filtering, for complete record-keeping
df.to_csv('decision_analysis.csv', index=False)

# Print the DataFrame
print(df)

# Get counts of each outcome
print("\n--- Outcome Counts ---")
print(df['winner'].value_counts())
