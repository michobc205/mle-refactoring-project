# King County House Prices: Refactoring Project

The original notebook `King-County.ipynb` cleans the King County house sales data, builds new features and trains a price model, all in one long notebook.
This project **refactors** it: the data cleaning and the feature engineering become **tested functions in Python files**, joined in one **reusable pipeline**.
The task itself is described in [`project-description.md`](project-description.md).

## What is in this repository

| File or folder | What it is |
|---|---|
| `King-County.ipynb` | The original notebook (unchanged) |
| `King-County-refactored.ipynb` | The refactored copy: it calls the functions below. The modeling section is unchanged. |
| `cleaning.py` | Data cleaning functions (`clean_data` runs them all) |
| `features.py` | Feature engineering functions (`add_features` runs them all) |
| `pipeline.py` | The reusable pipeline: `run_pipeline()` and `split_features_and_target()` |
| `tests/` | 24 tests for the functions and the pipeline |
| `bonus_solution/` | Reference material of the template for the optional tasks (not part of my work) |
| `data/` | The King County data set |

## How to run it

```bash
uv sync --locked                 # create the environment (Python 3.13.17, needs uv 0.13.5 or newer)
uv run pytest                    # run the 24 tests
uv run python pipeline.py        # run the pipeline as a program
```

For the notebook, open `King-County-refactored.ipynb` and choose the kernel of the project's `.venv`.

## What the functions do

| Step | Function | Result |
|---|---|---|
| Cleaning | `remove_implausible_houses` | Removes the house with 33 bedrooms, by a condition (not by a row number) |
| Cleaning | `fix_basement_size` | `sqft_basement` becomes `sqft_living - sqft_above`, so the `?` values disappear |
| Cleaning | `fill_missing_view_and_waterfront` | Missing values become 0 |
| Cleaning | `add_last_known_change` | Renovation year if renovated, else the build year |
| Features | `add_sqft_price` | Price per square foot (it uses the price, so it is never a model input) |
| Features | `add_center_distance` | Distance in km to the center of wealth |
| Features | `add_water_distance` | Distance in km to the nearest waterfront house (one numpy calculation instead of two loops) |

The waterfront reference points are computed **once** (`get_waterfront_locations`) and can be reused for new houses.

## Results

The tests compare the refactored code with the numbers of the original notebook:

| Check | Result |
|---|---|
| Rows after cleaning | 21,596 (one house removed), no missing values |
| Mean `last_known_change` | 1972.946 |
| Mean `sqft_price` | 59.7286 |
| Mean `center_distance` | 17.5333 km |
| Mean `water_distance` | 5.544 km (0 for all waterfront houses) |

The model scores of the original and the refactored notebook are **identical** (tested in the same environment): adjusted R² 0.43, 0.48 and 0.75 for the three linear models, and **test R² 0.844** for the regularized model.

## Checklist

**Core tasks**

- [x] I read and understood the notebook.
- [x] I made a copy of the notebook for refactoring it.
- [x] I removed unnecessary code and only kept relevant code for data cleaning and feature engineering.
- [x] I transformed the steps in the Data Preparation and Feature Engineering sections into functions.
- [x] Functions are in a separate Python file.

**Optional tasks**

- [ ] I refactored the modeling step and added it to the pipeline. *(The modeling stays in the notebook.)*
- [ ] I built a FastAPI app that allows me to create, read, update and delete houses.
- [ ] I created a `Dockerfile` and a `docker-compose.yaml` file for the app.
- [ ] I built the container with Docker Compose and I am able to run the app inside of it.
- [x] I created tests for my functions.

## Answers to the three questions

### 1. What steps did you take to complete the project?

1. I read the project description and the notebook, and listed what it does: cleaning, feature engineering, modeling.
2. I ran the data preparation on the real data and wrote down the reference numbers (rows, means) to use them as tests.
3. I moved the cleaning steps into `cleaning.py`, one function per step, with tests first.
4. I moved the feature steps into `features.py`, and replaced the loops with numpy.
5. I joined everything in `pipeline.py` and tested it, also for a new house without a price.
6. I made a copy of the notebook that calls the functions and keeps the modeling section.
7. I ran the original and the refactored notebook in the same environment and compared the model scores: they are identical.

### 2. What challenges did you face?

- **A fixed row number.** The notebook removed the 33-bedroom house with `drop(15856)`. I replaced it with a condition, so it does not depend on the row order.
- **Slow loops.** The water distance was built with two loops and about 3 million single calculations. In a test run the original notebook needed about 2 minutes, the refactored one about 10 seconds. One numpy calculation gives the same numbers.
- **Reusing the features for new houses.** The water distance needs the waterfront houses of the training data, and `sqft_price` needs a price that a new house does not have. I separated the reference points from the calculation and made `sqft_price` optional.
- **Proving that nothing changed.** I used the numbers of the original notebook as tests, and compared the model scores of both notebooks.
- **Library versions.** The polynomial model without regularization gave 0.87 in one environment and 0.75 in another, because it is sensitive to the library versions. Only the same environment gives a fair comparison.
- **Text that did not match the results.** Some sentences of the notebook said "price per square foot" for the target (it is the price), "a little further" for a jump of 0.27, and "83%" for a test R² of 0.844. I corrected them in the copy.

### 3. What would you do differently if you had more time?

- Move the **modeling step** into the pipeline (a scikit-learn pipeline with a transformer that remembers the waterfront reference points), and save it.
- Build the **FastAPI app** with a database, a **Dockerfile** and Docker Compose, and test it with the container running.
- Add an **automatic test run** (GitHub Actions) for every push.
- Split the data into training and test sets **before** computing the water distance, so that the reference points come from the training data only.
- Check the input of new houses with Pydantic, and add tests for more edge cases.

## Tools

Python 3.13, pandas, NumPy, scikit-learn, pytest and uv. I used Claude as a coding assistant for the plan, the code and the tests.
