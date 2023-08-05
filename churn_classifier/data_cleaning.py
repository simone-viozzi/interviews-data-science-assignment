import pandas as pd


def clean_last_new_job(col: pd.Series):
    return (
        col.fillna(0)
        .pipe(
            lambda x: x.replace(
                {
                    "never": 0,
                    ">4": 5,
                },
            ),
        )
        .astype("int")
    )


def clean_experience(col: pd.Series):
    return (
        col.fillna(0)
        .pipe(
            lambda x: x.replace(
                {
                    "<1": 0,
                    ">20": 21,
                },
            ),
        )
        .astype("int")
    )


def clean_company_size(col: pd.Series):
    return (
        col.fillna(0)
        .pipe(
            lambda x: x.replace(
                {
                    "<10": 10,
                    "10/49": 49,
                    "50-99": 99,
                    "100-500": 499,
                    "500-999": 999,
                    "1000-4999": 4999,
                    "5000-9999": 9999,
                    "10000+": 10000,
                },
            ),
        )
        .astype("int")
    )


def clean_education_level(col: pd.Series):
    return (
        col.fillna(0)
        .pipe(
            lambda x: x.replace(
                {
                    "Primary School": 1,
                    "High School": 2,
                    "Graduate": 3,
                    "Masters": 4,
                    "Phd": 5,
                },
            ),
        )
        .astype("int")
    )


def clean_dataset(df: pd.DataFrame):
    return df.assign(
        last_new_job=clean_last_new_job(df.last_new_job),
        experience=clean_experience(df.experience),
        company_size=clean_company_size(df.company_size),
        education_level=clean_education_level(df.education_level),
        major_discipline=df.major_discipline.fillna("Unknown").astype("category"),
        city=df.city.astype("category"),
        gender=df.gender.fillna("Other").astype("category"),
        relevent_experience=df.relevent_experience.replace(
            {
                "Has relevent experience": True,
                "No relevent experience": False,
            },
        ).astype("bool"),
        enrolled_university=df.enrolled_university.fillna("Unknown").astype("category"),
        company_type=df.company_type.fillna("Other").astype("category"),
    )
