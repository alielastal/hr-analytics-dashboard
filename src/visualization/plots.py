# File: src/visualization/plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Set consistent style
plt.style.use('default')
sns.set_palette("Set2")
rcParams['figure.figsize'] = (10, 6)

def set_style():
    """Set consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("Set2")
    rcParams['figure.figsize'] = (10, 6)
    rcParams['font.size'] = 12

def plot_attrition_by_department(df, save_path=None):
    """Plot attrition rates by department"""
    set_style()
    
    # Calculate attrition rates by department
    dept_attrition = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()
    dept_attrition = dept_attrition['Yes'] * 100 if 'Yes' in dept_attrition.columns else pd.Series(0, index=dept_attrition.index)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Countplot
    sns.countplot(data=df, x='Department', hue='Attrition', ax=ax1)
    ax1.set_title('Employee Count by Department and Attrition')
    ax1.tick_params(axis='x', rotation=45)
    
    # Percentage plot
    dept_attrition.sort_values(ascending=False).plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Attrition Rate by Department (%)')
    ax2.set_ylabel('Attrition Rate (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_attrition_by_gender(df, save_path=None):
    """Plot attrition rates by gender"""
    set_style()
    
    # Calculate attrition rates by gender
    gender_attrition = df.groupby('Gender')['Attrition'].value_counts(normalize=True).unstack()
    gender_attrition = gender_attrition['Yes'] * 100 if 'Yes' in gender_attrition.columns else pd.Series(0, index=gender_attrition.index)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Countplot
    sns.countplot(data=df, x='Gender', hue='Attrition', ax=ax1)
    ax1.set_title('Employee Count by Gender and Attrition')
    
    # Percentage plot
    gender_attrition.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Attrition Rate by Gender (%)')
    ax2.set_ylabel('Attrition Rate (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_salary_vs_attrition(df, save_path=None):
    """Plot salary distributions by attrition status"""
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    sns.boxplot(data=df, x='Attrition', y='Salary', ax=ax1)
    ax1.set_title('Salary Distribution by Attrition Status')
    
    # Violin plot
    sns.violinplot(data=df, x='Attrition', y='Salary', ax=ax2)
    ax2.set_title('Salary Distribution (Violin Plot)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_age_vs_attrition(df, save_path=None):
    """Plot age distributions by attrition status"""
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    sns.histplot(data=df, x='Age', hue='Attrition', multiple='stack', ax=ax1)
    ax1.set_title('Age Distribution by Attrition Status')
    
    # KDE plot
    sns.kdeplot(data=df[df['Attrition'] == 'Yes'], x='Age', label='Left', ax=ax2, fill=True)
    sns.kdeplot(data=df[df['Attrition'] == 'No'], x='Age', label='Stayed', ax=ax2, fill=True)
    ax2.set_title('Age Distribution Comparison')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_correlation_heatmap(df, save_path=None):
    """Plot correlation heatmap for numerical features"""
    set_style()
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_df = df[numerical_cols]
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Features')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_education_attrition(df, save_path=None):
    """Plot attrition by education level"""
    set_style()
    
    # Map education codes to meaningful labels if needed
    education_map = {
        1: 'High School',
        2: 'Bachelors', 
        3: 'Masters',
        4: 'PhD'
    }
    
    df_plot = df.copy()
    if 'Education' in df_plot.columns:
        df_plot['Education_Label'] = df_plot['Education'].map(education_map)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'Education_Label' in df_plot.columns:
        education_attrition = df_plot.groupby('Education_Label')['Attrition'].value_counts(normalize=True).unstack()
        if 'Yes' in education_attrition.columns:
            (education_attrition['Yes'] * 100).plot(kind='bar', ax=ax, color='steelblue')
    
    ax.set_title('Attrition Rate by Education Level (%)')
    ax.set_ylabel('Attrition Rate (%)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_jobrole_analysis(df, save_path=None):
    """Plot analysis of job roles and attrition"""
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Attrition by job role (count)
    sns.countplot(data=df, y='JobRole', hue='Attrition', ax=ax1)
    ax1.set_title('Attrition by Job Role (Count)')
    ax1.legend(loc='lower right')
    
    # Attrition rate by job role (%)
    jobrole_attrition = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack()
    if 'Yes' in jobrole_attrition.columns:
        (jobrole_attrition['Yes'] * 100).sort_values(ascending=True).plot(
            kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Attrition Rate by Job Role (%)')
    ax2.set_xlabel('Attrition Rate (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig