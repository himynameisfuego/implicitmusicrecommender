from pathlib import Path
from typing import Tuple, List
import scipy
import pandas as pd
import implicit
import implicit.recommender_base
import scipy.sparse

USER_ARTISTS = "C:\Datasets\hetrec2011-lastfm-2k\\user_artists.dat"
ARTISTS = "C:\Datasets\hetrec2011-lastfm-2k\\artists.dat"

def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    """Load user artists file, return user-artists matrix"""
    user_artists = pd.read_csv(user_artists_file, sep="\t")
    user_artists.set_index(["userID", "artistID"], inplace=True)
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()      # compressed sparse row matrix 

class ArtistRetriever:
    """Get artist name from artist ID"""
    
    def __init__(self):
        self._artists_df = None
        
    def get_artist_name_from_id(self, artist_id: int) -> str:
        return self._artists_df.loc[artist_id, "name"]
    
    def load_artists(self, artists_file: Path) -> None:
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df



class ImplicitRecommender:
    """Uses Implicit library to generate user-specific recommendation via collaborative filtering"""
    
    def __init__(self, artist_retriever: ArtistRetriever, implicit_model: implicit.recommender_base.RecommenderBase):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model
        
    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix):
        self.implicit_model.fit(user_artists_matrix)
        
    def recommend(self, user_id: int, user_artists_matrix: scipy.sparse.csr_matrix, n: int) -> Tuple[List[str], List[float]]:
        artist_ids, scores = self.implicit_model.recommend(user_id, user_artists_matrix[n], n)
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id) for artist_id in artist_ids
        ]
        return artists, scores
        
if __name__ == "__main__":
    
    print("If encountering an error with OpenBLAS, set environmental virable OPENBLAS_NUM_THREADS=1 in command line")
    
    user_artists = load_user_artists(Path(USER_ARTISTS))
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path(ARTISTS))
    
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=100, iterations=20, regularization=0.01
    )
    
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    
    user_id = 10
    n_preferences = 2
    artists, scores = recommender.recommend(user_id=user_id, user_artists_matrix=user_artists, n=n_preferences)
    
    print("")
    print(f"Suggesting {n_preferences} preferences for user n.{user_id}:")
    for artist, score in zip(artists, scores):
        print(f"{artist} [Score: {score}]")