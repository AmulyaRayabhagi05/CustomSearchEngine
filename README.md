Instructions for running program

Dependencies:
pip install spacy
python -m spacy download en_core_web_sm

Creating and activating venv:
python3 -m venv venv
source venv/bin/activate

Necessary files (Already in folder):
wget.download("https://www.gutenberg.org/files/1661/1661-0.txt","book.txt")
wget http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
tar -xzf MovieSummaries.tar.gz

Running Parts 1 and 2
spark-submit pyspark_word_count.py
spark-submit pyspark_search_engine.py

Output Files are included in folder:
part1_output.txt
part2_output.txt