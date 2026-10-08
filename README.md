Instructions for running program

Dependencies:
pip install spacy
python -m spacy download en_core_web_sm

Creating and activating venv:
python3 -m venv venv
source venv/bin/activate

Necessary files:
wget.download("https://www.gutenberg.org/files/1661/1661-0.txt","book.txt")
wget http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
tar -xzf MovieSummaries.tar.gz

Running Parts 
spark-submit pyspark_search_engine.py

Output File(included in folder):
part2_output.txt
