import re
import string
import joblib
from spellchecker import SpellChecker
import pytesseract
from PIL import Image
from contractions import CONTRACTION_MAP

model = joblib.load('model.pkl')

classifier = model['toxic_classifier']
tfidf = model['tfidf']



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    with Image.open(image_path) as img:
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text



def expand_contraction(text):
    for key,val in CONTRACTION_MAP.items():
        text = text.lower()
        text = re.sub(r'{}'.format(key),val,text)
        
    return text



def spell_check(text):
    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    
    corrected_text = []
    for word in words:
        if word in misspelled:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word)
        else:
            corrected_text.append(word)
    
    return ' '.join(corrected_text)



def clean_text(text):
    text = str(text)
    
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.lower()
    text = re.sub("\n", " ", text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    
    return text.strip()



def toxicity_prediction(sample_script):
  temp = tfidf.transform([sample_script]).toarray()
  return classifier.predict_proba(temp)[:,1]