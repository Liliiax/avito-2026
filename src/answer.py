from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
OUTPUT_PATH = ROOT_DIR / "answer.csv"



# параметры: 

TOP_K = 50 # Сколько необходимо кандидатов в итоговом ответе

WORD_TOP_K = 500 # Сколько кандидатов берём из каждого TF-IDF поиска
CHAR_TOP_K = 500

QUERY_BATCH_SIZE = 100 # Размер батча запросов

WORD_MAX_FEATURES = 250_000 # TF-IDF параметры
CHAR_MAX_FEATURES = 180_000
RRF_K = 60



def build_texts(df, columns):

    """
    Склеивает несколько текстовых колонок в одну строку.

    """

    columns = [c for c in columns if c in df.columns]
    result = df[columns[0]].fillna("").astype(str)
    for col in columns[1:]:
        result = result.str.cat(
            df[col].fillna("").astype(str), sep=" ")
        
    return (result.str.replace(r"\s+", " ", regex=True).str.strip())



def get_top_k_sparse(matrix, k):

    """
    matrix: sparse matrix shape [n_items, n_queries]
    Для каждого query возвращает top-k item indices
    и соответствующие scores.

    """

    matrix = matrix.tocsc()
    result = []

    for query_idx in range(matrix.shape[1]):

        start = matrix.indptr[query_idx]
        end = matrix.indptr[query_idx + 1]
        item_indices = matrix.indices[start:end]
        scores = matrix.data[start:end]

        if len(scores) == 0:
            result.append((np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
            continue

        if len(scores) > k:

            keep = np.argpartition(scores, -k)[-k:] # оставляем k лучших и сортируем только их
            keep = keep[np.argsort(scores[keep])[::-1]]

            item_indices = item_indices[keep]
            scores = scores[keep]

        else:

            keep = np.argsort(scores)[::-1]
            item_indices = item_indices[keep]
            scores = scores[keep]

        result.append((item_indices.astype(np.int32), scores.astype(np.float32)))

    return result



def fuse_candidates(word_indices, word_scores, char_indices, char_scores, top_k=50, rrf_k=60):

    """
    Объединяем результаты Word TF-IDF и Char TF-IDF через Reciprocal Rank Fusion.
    RRF score: 1 / (k + rank)

    """

    scores = {}

    for rank, item_idx in enumerate(word_indices, start=1):

        rrf_score = 1.0 / (rrf_k + rank)
        scores[item_idx] = scores.get(item_idx, 0.0) + rrf_score

    for rank, item_idx in enumerate(char_indices, start=1):

        rrf_score = 0.8 / (rrf_k + rank)
        scores[item_idx] = scores.get(item_idx, 0.0) + rrf_score

    if not scores:
        return []

    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]  # аналогично берем top-k
    return [item_idx for item_idx, _ in best]


def main():

    

    # 1. read data:

    queries_df = pd.read_parquet(QUERIES_PATH) # берём только необходимые колонки benchmark_items.
    items_df = pd.read_parquet(ITEMS_PATH, columns=["item_id", "item_title_raw", "item_infm_params_text",])

    # 2. build item text

    item_texts = build_texts(items_df, ["item_title_raw", "item_infm_params_text", ])

    query_texts = build_texts(queries_df, ["search_query", "search_infm_params_text",])

    # 3. word TF-IDF

    word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=WORD_MAX_FEATURES, sublinear_tf=True,  dtype=np.float32,)
    item_word_matrix = word_vectorizer.fit_transform(item_texts)

    # 4.char TF-IDF

    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=CHAR_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,)
    item_char_matrix = char_vectorizer.fit_transform(item_texts)


    del item_texts # после построения матриц исходные строки больше не нужны

    # 5. поиск

    n_queries = len(queries_df)
    answers = []

    for batch_start in range(0, n_queries, QUERY_BATCH_SIZE):

        batch_end = min(batch_start + QUERY_BATCH_SIZE, n_queries)
        batch_queries = query_texts.iloc[batch_start:batch_end]

        word_query_matrix = word_vectorizer.transform(batch_queries)
        word_scores_matrix = (item_word_matrix @ word_query_matrix.T)

        word_results = get_top_k_sparse(word_scores_matrix, WORD_TOP_K)

        char_query_matrix = char_vectorizer.transform(batch_queries)

        char_scores_matrix = (item_char_matrix @ char_query_matrix.T)

        char_results = get_top_k_sparse(char_scores_matrix, CHAR_TOP_K)


        for local_idx in range(batch_end - batch_start):

            word_indices, word_scores = (word_results[local_idx])

            char_indices, char_scores = (char_results[local_idx])

            candidate_indices = fuse_candidates(word_indices, word_scores, char_indices, char_scores,

                top_k=TOP_K, rrf_k=RRF_K,)


            item_ids = [str(items_df.iloc[idx]["item_id"]) for idx in candidate_indices] # переводим индексы в item_id

            item_ids = list(dict.fromkeys(item_ids)) # защита от дублей
 
            answers.append(item_ids)


    result = pd.DataFrame({

            "query_id": queries_df["query_id"].values,
            "answer": [ " ".join(x) for x in answers], })

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":

    main() 