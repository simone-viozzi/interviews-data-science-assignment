# Data Science Interview Assignment

## Introduction

If you read this file, you have passed our initial screening. Well done! :clap: :clap: :clap:

:rocket: The next step to join the Data Science team of [xtream](https://xtreamers.io) is this assignment.
You will find several datasets: please choose **only one**.
For each dataset, we propose several challenges. You **do not need to
complete all of them**, but rather only the ones you feel comfortable about or the ones that interest you.

:sparkles: Choose what really makes you shine!

:watch: The deadline for submission is **10 days** after you are provided with the link to this repository, so that you can move at your own pace.

:heavy_exclamation_mark: **Important**: you might feel the tasks are too broad, or the requirements are not
fully elicited. **This is done on purpose**: we wish to let you take your own way in
extracting value from the data and in developing your own solutions.

### Deliverables

Please fork this repository and work on it as if you were taking on a real-world project.
On the deadline, we will check out your work.

:heavy_exclamation_mark: **Important**: At the end of this README, you will find a blank "How to run" section.
Please write there instructions on how to run your code.

### Evaluation

Your work will be assessed according to several criteria, for instance:

* Work Method
* Understanding of the business problem
* Understanding of the data
* Correctness, completeness, and clarity of the results
* Quality of the codebase
* Documentation

:heavy_exclamation_mark: **Important**: this is not a Kaggle competition, we do not care about model performance.
No need to get the best possible model: focus on showing your method and why you would be able to get there,
given enough time and support.

---

### Diamonds

**Problem type**: regression

**Dataset description**: [Diamonds readme](./datasets/diamonds/README.md)

Don Francesco runs a jewelry. He is a very rich fellow, but his past is shady: be sure not to make him angry.
Over the years, he collected data from 5000 diamonds.
The dataset provides physical features of the stones, as well as their value, as estimated by a respected expert.

#### Challenge 1

**Francesco wants to know which factors influence the value of a diamond**: he is not an expert, he wants simple and clear messages.
However, he trusts no one, and he hired another data scientist to get a second opinion on your work.
Create a Jupyter notebook to explain what Francesco should look at and why.
Your code should be understandable by a data scientist, but your text should be clear for a layman.

#### Challenge 2

Then, Francesco tells you that the expert providing him with the stone valuations disappeared.
**He wants you to develop a model to predict the value of a new diamond given its characteristics**.
He insists on a point: his customer are not easy-going, so he wants to know why a stone is given a certain value.
Create a Jupyter notebook to meet John's request.

#### Challenge 3

Francesco likes your model! Now he wants to use it. To improve the model, Francesco is open to hire a new expert and
let him value more stones.
**Create an automatic pipeline capable of training a new instance of your model from the raw dataset**.

#### Challenge 4

Finally, Francesco wants to embed your model in a web application, to allow for easy use by his employees.
**Develop a REST API to expose the model predictions**.

---

### Italian Power Load

**Problem type**: time series forecasting

**Dataset description**: [Power Load readme](./datasets/italian-power-load/README.md)

It is your first day in the office and your first project is about time series forecasting.
Your customer is Zap Inc, an imaginary Italian utility: they will provide you with the daily Italian Power Load from 2006 to 2022.
Marta, a colleague of yours, provides you with a wise piece of advice: be careful about 2020, it was a very strange year...

#### Challenge 1

Zap Inc asks you for a complete report about the main feature of the power load series.
**Create a Jupyter notebook to answer their query.**

#### Challenge 2

Then, your first forecasting model.
**You are asked to develop a long-term model to predict the power load 1 year ahead.**
Disregard 2020, 2021, and 2022: use 2019 as test.

#### Challenge 3

Long-term was great, but what about short term?
**Your next task is to create a short-term model to predict the power load 1 day ahead.**
Disregard 2020, 2021, and 2022: use 2019 as test.

#### Challenge 4

Finally, production trial.
**Pick one of your models and develop and end-to-end pipeline to train and evaluate it on 2020 and 2021.**

#### Challenge 5

Zap Inc is not impressed by the performance of your model in 2020. You should defend your results.
**Create a notebook to comment and explain the performance of your model in 2020.**

---

### Employee Churn

**Problem type**: classification

**Dataset description**: [Employee churn readme](./datasets/employee-churn/README.md)

You have just been contracted by Pear Inc, a multinational company worried about its poor talent retention.
In the past few months, they collected data about their new employees. All of them come from classes
the company is sponsoring, yet many enter Pear just to leave a few months later.
This is a huge waste of time and money.

The HR department of the company wants you to understand what is going on and to prevent further bleeding.

#### Challenge 1

Pear Inc wants you to understand what are the main traits separating the loyal employees from the others.
**Create a Jupyter notebook to answer their query.**

#### Challenge 2

Then, a predicting model.
**You are asked to create a model to predict whether a new employee would churn**.
If possible, the company wants to know the likelihood of the churn.

#### Challenge 3

Wow, the model works great, but why does it?
**Try and make the model interpretable**, by highlighting the most important features and how each prediction is made.

#### Challenge 4

Now, production trial.
**Develop and end-to-end pipeline to train a model given a new dataset.**
You can assume that the new dataset has exactly the same structure as the provided one:
possible structural changes will be managed by your fellow data engineers.

#### Challenge 5

Finally, Pear Inc is happy with your results!
Now they want to embed your model in a web application.
**Develop a REST API to expose the model predictions**.

---

## How to run

To run the app:

```bash
docker compose up --build
```

The API docs will be available at `http://localhost:8000/docs`.
