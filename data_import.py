import pandas as pd
from textacy import preprocess_text


def import_data(filepath):
    df = pd.read_excel(filepath, header=0, sheet_name='FullExtract')

    df['year'] = df.DateOfReceipt.astype('str').str[0:4]

    # We will subset the df to only include key values
    df = df[['ItemID', 'Summary', 'ORG_UNIT_NAME', 'year']].copy()
    df = df[df.Summary.str.match(
        'To ask the Secretary of State for Transport')].copy()

    # Remove the leading sentence
    df['Summary'] = df.Summary.str.replace(
        '^To ask the Secretary of State for Transport, ', '')
    return (df)


def clean_text(df):
    '''
    Func to clean up textual data to remove items that aren't likely to be
    useful in machine learning.
    '''

    def preprocess_text_settings(string_in):
        string_in = preprocess_text(
            string_in,
            fix_unicode=True,
            lowercase=True,
            no_urls=True,
            no_emails=True,
            no_numbers=True,
            no_accents=True,
            no_punct=True)
        return (string_in)

    df['summary_cleaned'] = df['Summary'].apply(preprocess_text_settings)
    return (df)


def import_and_clean(filepath='data/Extract.xlsx'):
    df = import_data(filepath)
    df = clean_text(df)
    return (df)
