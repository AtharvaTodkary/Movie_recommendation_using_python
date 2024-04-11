import time
import streamlit as st
import pandas as pd
import numpy as np
import difflib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('final_data.csv')
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(data['tag']).toarray()
similarity = cosine_similarity(vectors)
moviesList = data['original_title'].values.tolist()

def content_based_recommend(movie_name):
    try:
        # Finding the closest match based on common words
        max_common_words = -1
        closest_match = None
        for title in moviesList:
            common_words = set(movie_name.lower().split()) & set(title.lower().split())
            if len(common_words) > max_common_words:
                max_common_words = len(common_words)
                closest_match = title

        # If no common words found, consider it as no match
        if max_common_words == 0:
            print("No close match found for the movie name.")
            return [], movie_name

        # Finding the movie index
        movie_index = moviesList.index(closest_match)

        # Getting cosine distance
        cosine_distance = similarity[movie_index]

        # Sorting based on similarity
        required_movies = sorted(enumerate(cosine_distance), reverse=True, key=lambda x: x[1])[:20]

        # Retrieving movie details
        movies = []

        for i in required_movies:
            item = []
            imdb_id = data.iloc[i[0]]['imdb_id']
            item.append(data[data['imdb_id'] == imdb_id]['original_title'].values[0])
            item.append(data[data['imdb_id'] == imdb_id]['wiki_link'].values[0])
            item.append(data[data['imdb_id'] == imdb_id]['poster_path'].values[0])
            movies.append(item)

        return movies, closest_match

    except Exception as e:
        print("An error occurred:", str(e))
        return [], movie_name

st.set_page_config(layout="wide")
st.title("HYBRID MOVIE RECOMMENDATION SYSTEM :popcorn::popcorn::popcorn:")
flag1 = 0
with st.form('recommendMovies'):
    user_id = st.text_input("ENTER USER ID")
    movieName = st.text_input("ENTER MOVIE NAME")
    rating = st.slider("RATE THE MOVIE", min_value=1, max_value=5, step=1)
    if st.form_submit_button("SUBMIT"):
        flag1 = 1
        movies, searched_movie = content_based_recommend(movieName)
        if movies:
            st.write(f"User ID: {user_id}")
            st.write(f"Searched Movie: {searched_movie}")
            st.write(f"Rating given: {rating}")
            st.title("Movies recommended for you")
            index = 0
            while index < 9:
                col1, col2, col3 = st.columns(3)

                # Display information for the first movie
                col1.markdown(f"<h5>{movies[index][0]}</h5>", True)
                try:
                    col1.image(movies[index][2], width=200)
                except Exception as e:
                    col1.image("https://images.unsplash.com/photo-1509281373149-e957c6296406?q=80&w=1456&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=200)
                # col1.markdown(f"<a href = '{movies[index][1]}'>Click</a>", True)

                # Increment index
                index += 1

                # Display information for the second movie
                col2.markdown(f"<h5>{movies[index][0]}</h5>", True)
                try:
                    col2.image(movies[index][2], width=200)
                except Exception as e:
                    col2.image("https://images.unsplash.com/photo-1509281373149-e957c6296406?q=80&w=1456&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=200)
                # col2.markdown(f"<a href = '{movies[index][1]}'>Click</a>", True)

                # Increment index
                index += 1

                # Display information for the third movie
                col3.markdown(f"<h5>{movies[index][0]}</h5>", True)

                try:
                    col3.image(movies[index][2], width=200)
                except Exception as e:
                    col3.image(
                        "https://images.unsplash.com/photo-1509281373149-e957c6296406?q=80&w=1456&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=200)
                # col3.markdown(f"<a href = '{movies[index][1]}'>Click</a>", True)

                # Increment index
                index += 1

            # Store user search information in user_history.csv
            user_history = pd.DataFrame({'user_id': [user_id],
                                         'movie_title': [searched_movie],
                                         'rating': [rating]})
            user_history.to_csv('user_history.csv', mode='a', header=False, index=False)
        else:
            st.error("SEARCHED MOVIE NOT FOUND !!!")
