import pandas as pd
import seaborn as sns
from config import config
import matplotlib.pyplot as plt

def sample_dataframe(df: pd.DataFrame, max_rows: int = 100_000) -> pd.DataFrame:
    if len(df) > max_rows:
        print(f"Data too large ({len(df)} rows). Sampling {max_rows} rows for analysis.")
        df_sampled = df.sample(n=max_rows, random_state=config.SEED)
        return df_sampled
    else:
        return df

def basic_info(df: pd.DataFrame):
    print("=" * 50)
    print("Basic Dataset Information:")
    print(df.info())

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nMissing Values Summary:")
    print(df.isnull().sum())
    print("=" * 50)

def plot_univariate(df: pd.DataFrame, max_cat_unique: int = 20):
    numeric_cols = df.select_dtypes(include=['number']).columns
    category_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True, bins=20)
        plt.title(f'Histogram of {col}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    for col in category_cols:
        if df[col].nunique() <= max_cat_unique:
            plt.figure(figsize=(10, 5))
            sns.countplot(y=col, data=df, order=df[col].value_counts().index)
            plt.title(f'Bar Chart of {col}')
            plt.tight_layout()
            plt.show()

def plot_bivariate(df: pd.DataFrame, target_col=config.TARGET, max_cat_unique: int = 20):
    numeric_cols = df.select_dtypes(include=['number']).columns.drop(target_col, errors='ignore')
    category_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[target_col], y=df[col])
        plt.title(f'{col} vs {target_col} (Boxplot)')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=df[col], y=df[target_col])
        plt.title(f'{col} vs {target_col} (Scatter)')
        plt.tight_layout()
        plt.show()

    for col in category_cols:
        if df[col].nunique() <= max_cat_unique:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=col, y=target_col, data=df, estimator='mean')
            plt.title(f'Mean {target_col} by {col}')
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.show()

def analyze_data(df: pd.DataFrame):
    print("==== [Step 1] Basic Information ====")
    df = sample_dataframe(df)
    basic_info(df)

    print("\n==== [Step 2] Univariate Analysis ====")
    plot_univariate(df)

    print("\n==== [Step 3] Bivariate Analysis ====")
    plot_bivariate(df)