import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title of the web app
st.title("Movie Recommendation System")

# Load the dataset
@st.cache_data
def load_data():
    # Replace the path with the correct path to your movies dataset
    movies_data = pd.read_csv(r'movies.csv', encoding='latin-1', low_memory=False)
    return movies_data

movies_data = load_data()



# Selecting relevant features for the recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Filling missing values with empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining the selected features into one
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Converting text data to feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Cosine similarity for feature vectors
similarity = cosine_similarity(feature_vectors)

# User input for the movie name
movie_name = st.text_input("Enter your favorite movie name:")

# Show recommendations based on user input
if movie_name:
    # List of all movie titles in the dataset, ensuring non-string entries are removed
    list_of_all_titles = movies_data['title'].dropna().astype(str).tolist()

    # Ensure movie_name is a string and not empty
    if isinstance(movie_name, str) and movie_name.strip():
        # Find close match for the movie name given by the user
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if find_close_match:
            close_match = find_close_match[0]

            # Find the index of the movie with the title
            index_of_the_movie = movies_data[movies_data.title == close_match].index[0]

            # Get a list of similar movies based on cosine similarity
            similarity_score = list(enumerate(similarity[index_of_the_movie]))

            # Sort the movies based on similarity score
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            # Show top 30 recommended movies in a table format
            movie_recommendations = []
            for i, movie in enumerate(sorted_similar_movies[:30], start=1):
                index = movie[0]
                title = movies_data.iloc[index]['title']
                genre = movies_data.iloc[index]['genres']
                overview = movies_data.iloc[index]['overview'] if 'overview' in movies_data.columns else "Not available"
                
                movie_recommendations.append({
                    'Movie Name': title,
                    'Genre': genre,
                    'Overview': overview
                })

            # Convert to DataFrame for display
            movie_recommendations_df = pd.DataFrame(movie_recommendations)
            
            # Display the table in Streamlit
            st.subheader("Movies suggested for you:")
            st.table(movie_recommendations_df)
            
        else:
            st.write("No close match found for your movie. Please try another movie.")
    else:
        st.write("Please enter a valid movie name.")
