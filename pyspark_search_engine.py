import math, re, spacy
from pyspark.sql import SparkSession

# Load spaCy stop words
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORDS = nlp.Defaults.stop_words

def tokenize(text):
    return [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if w not in STOPWORDS and len(w) > 1]

def main():

    spark = SparkSession.builder.appName("MovieSearch").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # 1. PySpark loads metadata 
    titles = sc.broadcast(sc.textFile("MovieSummaries/movie.metadata.tsv")
               .map(lambda l: (l.split('\t')[0], l.split('\t')[2]) if len(l.split('\t')) >= 3 else None)
               .filter(bool).collectAsMap())

    # 2. PySpark loads plot summaries
    plots = sc.textFile("MovieSummaries/plot_summaries.txt") \
              .map(lambda l: (l.split('\t', 1)[0], tokenize(l.split('\t', 1)[1])) if '\t' in l else None) \
              .filter(bool).cache()
    N = plots.count()

    # 3. PySpark MapReduce: Term Frequency (TF)
    tf = plots.flatMap(lambda x: [((x[0], t), 1) for t in x[1]]).reduceByKey(lambda a, b: a + b) \
              .map(lambda x: (x[0][1], (x[0][0], float(x[1]))))
    
    # 4. PySpark MapReduce: Document Frequency (DF) & IDF
    idf = plots.flatMap(lambda x: [(t, x[0]) for t in set(x[1])]).groupByKey() \
               .mapValues(lambda docs: math.log10(N / float(len(docs))))

    # 5. PySpark Joins TF and IDF -> TF-IDF
    tfidf = tf.join(idf).map(lambda x: (x[1][0][0], (x[0], x[1][0][1] * x[1][1]))).cache()
    
    # 6. PySpark Vector L2 Norms for Cosine Similarity
    doc_norms = sc.broadcast(tfidf.map(lambda x: (x[0], x[1][1]**2))
                  .reduceByKey(lambda a, b: a + b)
                  .mapValues(math.sqrt).collectAsMap())

    idf_map = idf.collectAsMap()
    term_idx = tfidf.map(lambda x: (x[1][0], (x[0], x[1][1]))).cache()

    # 7. Execute Queries inside PySpark
    with open("search_queries.txt") as f:
        queries = [q.strip() for q in f if q.strip()]

    print("\n" + "="*50 + "\nSEARCH RESULTS\n" + "="*50)
    for q in queries:
        tokens = tokenize(q)
        if not tokens: continue
        print(f"\n>>> Query: '{q}'")

        if len(tokens) == 1:
            # Single-term search using PySpark filtering
            res = term_idx.filter(lambda x: x[0] == tokens[0]).map(lambda x: x[1]).top(10, key=lambda x: x[1])
            for r, (doc, score) in enumerate(res, 1):
                print(f"  {r:2d}. {titles.value.get(doc, 'Unknown')} - Score: {score:.5f}")
        else:
            # Multi-term Cosine Similarity using PySpark RDD transformations
            q_tf = {t: float(tokens.count(t)) for t in tokens}
            q_tfidf = {t: tf * idf_map.get(t, 0.0) for t, tf in q_tf.items()}
            q_norm = math.sqrt(sum(v**2 for v in q_tfidf.values()))
            if q_norm == 0: continue

            q_b = sc.broadcast(q_tfidf)
            res = tfidf.filter(lambda x: x[1][0] in q_b.value) \
                       .map(lambda x: (x[0], x[1][1] * q_b.value[x[1][0]])) \
                       .reduceByKey(lambda a, b: a + b) \
                       .map(lambda x: (x[0], x[1] / (q_norm * doc_norms.value.get(x[0], 1.0)))) \
                       .top(10, key=lambda x: x[1])
            for r, (doc, sim) in enumerate(res, 1):
                print(f"  {r:2d}. {titles.value.get(doc, 'Unknown')} - Cosine Sim: {sim:.5f}")

    spark.stop()

if __name__ == "__main__":
    main()
