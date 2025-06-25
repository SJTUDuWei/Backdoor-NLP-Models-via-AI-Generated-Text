from scipy import spatial
from sentence_transformers import SentenceTransformer
import Levenshtein
import difflib


class Diversity():
    def __init__(self, diversity_ranker):
        if diversity_ranker == "levenshtein":
            self.ranker = LevenshteinRanker()
        if diversity_ranker == "euclidean":
            self.ranker = EuclideanRanker()
        if diversity_ranker == "diff":
            self.ranker = DiffRanker()

    def rank(self, input_phrase, para_phrases):
        return self.ranker.rank(input_phrase, para_phrases)


class LevenshteinRanker():
    def rank(self, input_phrase, para_phrases):        
        diversity_scores = []
        for para_phrase in para_phrases:              
            distance = Levenshtein.distance(input_phrase.lower(), para_phrase)
            diversity_scores.append((para_phrase, distance))
            
        diversity_scores.sort(key=lambda x:x[1], reverse=True)
        return diversity_scores


class EuclideanRanker():
    def __init__(self):
        self.diversity_model = SentenceTransformer('all-mpnet-base-v2', device='cuda', cache_folder='./models')

    def rank(self, input_phrase, para_phrases):
        diversity_scores = []
        phrases = [input_phrase.lower()] + [p.lower() for p in para_phrases]
        phrases_enc = self.diversity_model.encode(phrases, show_progress_bar=False)
        for i, para_phrase in enumerate(para_phrases):              
            euclidean_distance = (spatial.distance.euclidean(phrases_enc[0], phrases_enc[i+1]))
            diversity_scores.append((para_phrase, euclidean_distance))

        diversity_scores.sort(key=lambda x:x[1], reverse=True)
        return  diversity_scores
  

class DiffRanker():
    def rank(self, input_phrase, para_phrases):
        differ = difflib.Differ()
        diversity_scores = []

        for para_phrase in para_phrases:
            diff = differ.compare(input_phrase.split(), para_phrase.split())
            count = 0
            for d in diff:
                if "+" in d or "-" in d:
                    count += 1
            diversity_scores.append((para_phrase, count))

        diversity_scores.sort(key=lambda x:x[1], reverse=True)
        return diversity_scores