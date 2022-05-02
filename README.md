# COMP6630-FinalProject
Group 4's Movie Recommendation System Project for COMP 6630 Final Project

## Prerequesites

- Create a virtual environment:
```
py -3.7.8 -m venv .venv
```
- Activate virtual environment:
```
.venv/Scripts/activate.bat
```

- Install required packages:
```
python -m pip install -r requirements.txt
```

## Algorithms
- Here is how to run each of the aglorithms:

- First make sure you are in the `COMP6630-FinalProject` folder
```
cd COMP6630-FinalProject
```
- Run each command using the your desired python interpreter and then the pathname of each file

### Rotten Tomato Scores
- Baseline algorithm: `python RottenTomatoes/baselineRT.py`
- Decision Tree algorithm from scratch: `python RottenTomatoes/decisionTreeRT.py`
- Decision Tree algorithm using sckit-learn: `python RottenTomatoes/rottenTomatoDT.py`

### Box Office Revenue
- Baseline algorithm: `python box_office_revenue/baselineBoxOffice.py`
- Complex algorithms: `python box_office_revenue/box_office_revenue.py`

### Movie Recommendation System
- Baseline algorithm: `python KaggleMovies/baselineMovieRecommendation.py`
- KNN algorithm: `python "Movie Recommendation.py"`