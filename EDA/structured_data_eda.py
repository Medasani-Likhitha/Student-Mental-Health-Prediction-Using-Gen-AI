import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def structured_eda(df):

    # -------------------------
    # Data Cleaning
    # -------------------------
    df.columns = df.columns.str.strip()

    df['Your current year of Study'] = df['Your current year of Study'].str.extract('(\d+)').astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].mode()[0]).astype(int)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

    df.rename(columns={
        'Choose your gender': 'gender',
        'What is your course?': 'course',
        'Your current year of Study': 'year',
        'What is your CGPA?': 'cgpa',
        'Marital status': 'marital_status',
        'Do you have Depression?': 'depression',
        'Do you have Anxiety?': 'anxiety',
        'Do you have Panic attack?': 'panic_attack',
        'Did you seek any specialist for a treatment?': 'specialist_visit'
    }, inplace=True)

    # -------------------------
    # Create Dashboard Layout
    # -------------------------
    fig = plt.figure(figsize=(22, 14))

    fig.suptitle("Student Mental Health Analysis Dashboard", fontsize=22, fontweight='bold')

    # 1️⃣ Depression Across Years
    ax1 = plt.subplot(3, 3, 1)
    pd.crosstab(df['year'], df['depression']).plot(kind='bar', ax=ax1)
    ax1.set_title("Depression Across Years")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Count")

    # 2️⃣ Anxiety by Gender
    ax2 = plt.subplot(3, 3, 2)
    pd.crosstab(df['gender'], df['anxiety']).plot(kind='bar', ax=ax2)
    ax2.set_title("Anxiety by Gender")
    ax2.set_xlabel("Gender")

    # 3️⃣ Age Distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(df['Age'], bins=6)
    ax3.set_title("Age Distribution")
    ax3.set_xlabel("Age")

    # 4️⃣ Depression vs Anxiety
    ax4 = plt.subplot(3, 3, 4)
    pd.crosstab(df['depression'], df['anxiety']).plot(kind='bar', ax=ax4)
    ax4.set_title("Depression vs Anxiety")

    # 5️⃣ Anxiety vs CGPA
    ax5 = plt.subplot(3, 3, 5)
    pd.crosstab(df['cgpa'], df['anxiety']).plot(kind='bar', ax=ax5)
    ax5.set_title("Anxiety Impact on CGPA")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)

    # 6️⃣ Age vs Depression
    ax6 = plt.subplot(3, 3, 6)
    pd.crosstab(df['Age'], df['depression']).plot(kind='bar', ax=ax6)
    ax6.set_title("Age-wise Depression")

    # 7️⃣ Anxiety vs Marital Status
    ax7 = plt.subplot(3, 3, 7)
    pd.crosstab(df['marital_status'], df['anxiety']).plot(kind='bar', ax=ax7)
    ax7.set_title("Anxiety vs Marital Status")

    # 8️⃣ Correlation Heatmap (Manual using matplotlib)
    df_numeric = df.copy()
    target_cols = ['depression', 'anxiety', 'panic_attack', 'specialist_visit']
    for col in target_cols:
        df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})
    df_numeric['gender'] = df_numeric['gender'].map({'Female': 0, 'Male': 1})

    corr_matrix = df_numeric[['Age', 'gender', 'year', 'depression', 'anxiety', 'panic_attack']].corr()

    ax8 = plt.subplot(3, 3, 8)
    cax = ax8.matshow(corr_matrix)
    fig.colorbar(cax)
    ax8.set_xticks(range(len(corr_matrix.columns)))
    ax8.set_yticks(range(len(corr_matrix.columns)))
    ax8.set_xticklabels(corr_matrix.columns, rotation=45)
    ax8.set_yticklabels(corr_matrix.columns)
    ax8.set_title("Correlation Heatmap")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save as High Quality PNG
    png_path = Path.cwd() / "results" / "Student_Mental_Health_Dashboard.png"
    print(png_path)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.show()

