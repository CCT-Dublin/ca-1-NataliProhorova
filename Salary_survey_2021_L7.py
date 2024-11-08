#!/usr/bin/env python
# coding: utf-8

# df overview,  determine problems and key questions to perform task, rename columns,  dealing with duplicates , missed data , timestamp  and screen record test

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[2]:


mvp=["n.a.","?","NA","n/a", "na", "--"]


# In[3]:


df=pd.read_csv("Salary_Survey.csv", na_values=mvp)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


# rename titles for columns change text to number, number to relevant title
df.columns = range(len(df.columns))


# In[7]:


#lets replase numbers for text format to rename columns 
new_column_names = ["column_" + str(i) for i in range(len(df.columns))]
df.columns = new_column_names


# In[8]:


df.head(2)


# In[9]:


#rename columns 
df = df.rename(columns={"column_0":"Timestump","column_1": "Age","column_16":"Gender","column_15": "Highest_Degree"})


# In[10]:


df = df.rename(columns={"column_2":"Industry","column_3": "Job title","column_4":"Job context","column_5":"Annual Salary","column_6":"Bonus","column_7":"Currency","column_8":"Currency other","column_9":"Income context"})


# In[11]:


df = df.rename(columns={"column_10":"Country","column_11": "State","column_12":"City","column_13": "Work_Exp","column_14": "Rel_Exp"})


# In[12]:


non_null_counts = df.count()
print(non_null_counts)


# In[13]:


#no non-null values -all values in the columns 17-22 are missing or null,we can delete them
df.drop(columns=['column_17','column_18','column_19','column_20','column_21','column_22'], inplace=True)


# In[14]:


#detecting duplicate rows
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[15]:


# Calculate the sum of non-null values   for 

total_sum_non_null = duplicate_rows_df.count(axis=1).sum()
print("Total sum of non-null values:", total_sum_non_null)


# In[16]:


# Print duplicate rows with null values
print(duplicate_rows_df[duplicate_rows_df.isnull().any(axis=1)])


# In[17]:


#we can exclude now  93 duplicates rows ,as we did for columns before
df = df.iloc[:-93]



# In[18]:


#review 
df.tail(2)


# In[19]:


#review shape update
df.shape


# In[20]:


# identifying columns with missing data and deciding how to handle  missing values
print(df.isnull().sum())


# In[21]:


#replace missing values with (Unknown')/'NA'
#  for missed values 'City',replace with equal value from 'Country'
df['Industry'] = df['Industry'].fillna('Unknown')
df['Bonus']=df['Bonus'].fillna('0')
df['Country']=df['Country'].fillna('Unknown')
df['State']=df['State'].fillna('NA')
df['City']=df['City'].fillna(df['Country'])
df['Currency other']=df['Currency other'].fillna('NA')


   


# In[22]:


#replsce missing data with addutuinal column info/  fill remain with 'Unknown'
df['Job title'] = df['Job title'].fillna('Job context')
df['Job title'] = df['Job title'].fillna('Unknown')
df['Job context'] = df['Job context'].fillna('Unknown')




#  determine  target variable(“Annual salary” ), dropping irrelevant columns,  # Main target - annual salary, year mention in a title ,values in a Timestump nonsuffisient impact=drop , 'Job context'- most of the data missed , but  can replace missed data for Jod title and , categotical variable ' Income context' do not impact  numerical data in Target variable 
# 

# In[23]:


#take down non relevant to the target variable columns Timestamp ,'Income context'
# Main target - annual salary, year mention in a title , 
df.drop(columns=['Timestump','Income context'], inplace=True)


# In[24]:


## Clean data from ',', convert to integrets 
df['Annual Salary'] = df['Annual Salary'].str.replace(',','').astype(int)


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html     sort values , ascending=False)

# In[25]:


#find outliers , top 5 salaryes
df = df.sort_values('Annual Salary', ascending=False)

print(df['Annual Salary'].head(5))


# In[ ]:





# In[26]:


print(df[['Annual Salary','Bonus']].tail(5))


# to  handle missing values for Annual salary and bonus , I create a new variable 'Total Income' (sum of salary and bonus0 filter   1<>10 000    as total incime can not be less then 10 000, and after filtering the duplicate rows with  1 or 0 ,less then 10000   drop fro Annual salary 

# https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html   coerse   convert non-numeric values to NaN,  avoid potential errors 

# In[27]:


# Handle missing values ,determine outliers
df['Annual Salary'] = pd.to_numeric(df['Annual Salary'], errors='coerce')

#Descriptive statistics for Annual Salary  to find outliers
print(df['Annual Salary'].describe().astype(int))


# min  0    , -   we have a number of outliers , we can compare them to Bonus and Currency  ,before drop from DF .As salary represented by different currency, review what is a top value , check  if missing values in cureency -  we need to clean them from data   .For filter  i use Accending method https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html#pandas.DataFrame.sort_values

# In[28]:


#  Annual  salary 6000070000 CAD   can not be realistic, just  this value  has misssed - , replase with median 65000
# 6000070000 with 65000 in 'Annual Salary'
df.loc[df['Annual Salary'] == 6000070000, 'Annual Salary'] = 65000


# In[29]:


# Print  comdined df

print(df[['Annual Salary','Currency','Bonus']].head(10))


# In[30]:


print(df['Bonus'].dtype)


# Bonus  is additional to Annual salary  numerical category, no needs to drop missing duplicates: convert (df['Bonus'] to   integrets, replace missed data with'0 

# In[31]:


# Remove non-numeric characters and convert to numeric

df['Bonus'] = pd.to_numeric(df['Bonus'], errors='coerce')

# Replace missing values with 0
df['Bonus'].fillna(0, inplace=True)


# In[32]:


missing_values = df['Bonus'].isna().sum()
print(f"Number of missing values in 'Bonus': {missing_values}")


# In[33]:


df['Bonus'] = df['Bonus'].astype(int)


# most of the top salary   values represented by currensy tipe'Other' ,that make them only numbers - I filter all categorycal data fom 'Currency'  = 'Other'  and drop allrelated  rows 

# In[34]:


# most of outliers in Annual salary  without value ,we can  drop all rows were Currency 'Other'
# make   all upper 
df['Currency'] = df['Currency'].str.upper()


# In[35]:


# Identify duplicate rows where 'Currency' is 'OTHER'
duplicate_rows_df = df[df.duplicated(subset=['Currency'], keep=False) & (df['Currency'] == 'OTHER')]

# Drop duplicate rows
df = df.drop(duplicate_rows_df.index)


# In[36]:


#  determine unique categories  upgate 
df["Currency"].unique() 


# In[37]:


# Frequency table for 'Gender_distribution = df['Gender'].value_counts()
Currency_distribution = df['Currency'].value_counts()
print(Currency_distribution)


# In[38]:


#create new combined  Total income  

df['Total_Income'] = df['Annual Salary'] + df['Bonus']
df[['Annual Salary','Bonus','Total_Income','Currency']].tail(10)


# #by using combined variable I saved some informative data for main target 'Annual salary' 

# In[39]:


#assuming total income  can not be less then 10 000 , filter and drop  relevant rows  were value <=10000 , including 0 
df = df[df['Total_Income'] > 10000]


# In[40]:


#correlations between variables 
df[['Total_Income','Currency','Job title','Age','Country','Currency other','Highest_Degree','Industry']].head(6)


# current output below given as method demonstration, numbers  can not represent group correctly ,as valuees  for  currensy did not  converted  for one same currency ,for first look some outliers can be corrected (for Columbia) or take down as for row wit index 28021   no clear info given and currency USD /Rtice  can be misleading for furter analyse 

# In[41]:


df = df.drop(index=28021)
df.loc[3605, 'Currency'] = 'COP'


# In[42]:


#Descriptive statistics for Annual Salary (   
print(df['Total_Income'].describe().astype(int))


# 

# In[43]:


#assuming 171 missed values in a column "Gender",replase them with 'No Gender ' 

df['Gender'] = df['Gender'].fillna('No_Gender ')


# In[44]:


# identifying  unique values in a  column
df['Gender'].unique()


# In[45]:


#Replace with "NA" (No answer) non gender related  categories (missing values)  using the replace() method:
df['Gender'] = df['Gender'].replace(['Prefer not to answer', 'Other or prefer not to answer'], 'No_Gender')


# In[46]:


#  clean and standartise string
df['Gender'] = df['Gender'].str.upper()
df['Gender'] = df['Gender'].str.strip()


# In[47]:


# Frequency table for 'Gender_distribution = df['Gender'].value_counts()+ %  counts,0.2%
Gender_distribution = df['Gender'].value_counts(normalize=True) * 100
for gender, percentage in Gender_distribution.items():
    print(f"{gender}: {percentage:.2f}%")


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html    - Value counts/https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.Series.value_counts.html  (normalize=True)   divide each value on a total sum of all values ,simpe proportion to get persentage when *100/The round(1) used to return  the calculations  to  decimal

# In[48]:


#standartise "'highest degree" variables 
#replace missing values,rename all   confused values with most close unic realted names
#replace missing values with (Not applicable ')  "NA"
df['Highest_Degree'] = df['Highest_Degree'].fillna('NON_DEGREE')


# In[49]:


#  determine unique categories in Higest  Education column
array = df["Highest_Degree"].unique() 
print(array)


# In[50]:


#rename values
df['Highest_Degree'] = df['Highest_Degree'].replace(['Professional degree (MD, JD, etc.)','Master\'s degree' ], 'MD')


# In[51]:


#rename values
df['Highest_Degree'] = df['Highest_Degree'].replace(['Some college','College degree'], 'CD')


# In[52]:


#Assuming Higer School and NA  eqoal to no Proffesional degreee,create a new combined category non_degree after assign same name for Higer scholl as for missing valus 
df['Highest_Degree'] = df['Highest_Degree'].str.replace('High School','NON_DEGREE')
   


# In[53]:


# Determine the distribution of categorical variables 
# Frequency table for 'highest education'Education_distribution = df['highest education level'].value_counts()
Education_distribution = df['Highest_Degree'].value_counts()
print(Education_distribution)


# https://pandas.pydata.org/docs/user_guide/visualization.html   visualise Education distribution   

# In[54]:


# Assuming Education_distribution is our Series
plt.figure(figsize=(8, 4))
plt.pie(Education_distribution, labels=Education_distribution.index, autopct='%1.1f%%', startangle=160)
plt.title('Education_distribution')
plt.axis('equal') 
plt.show()


# #create a new variable "Demographics"   that could provide different approach to analising releation betveen "Gender" ,Education"   and Salary   

# In[55]:


# Filter for specific groups  Professional_Education
filtered_df = df[df['Highest_Degree'].isin(['CD', 'MD', 'PhD'])]

# Calculate the total count for filtered groups
filtered_count = filtered_df['Highest_Degree'].count()

print("Total count for Professional_Education:", filtered_count)


# In[56]:


# Calculating % distribution of degrees
degree_distribution = df['Highest_Degree'].value_counts(normalize=True) * 100
for degree, percentage in degree_distribution.items():
    print(f"{degree}: {percentage:.2f}%")


# In[57]:


#  'df' with columns 'Highest Education Level' and 'Gender'
df['Demographics'] = df['Highest_Degree'] + '_' + df['Gender']


# In[58]:


# identifying  categories/values in a new   column
df['Demographics'].unique()


# https://pandas.pydata.org/docs/reference/api/pandas.Series.str.strip.html#pandas.Series.str.strip       clean new string [Demographics]    from  whitespaces, standartise values names 

# In[59]:


#  clean and standartise string
df['Demographics'] = df['Demographics'].str.upper()
df['Demographics'] = df['Demographics'].str.strip()


# In[60]:


# Filter for specific groups  Woman_Education
filtered_df = df[df['Demographics'].isin(['CD_WOMAN', 'MD_WOMAN', 'PHD_WOMAN'])]
# Calculate the total count of all demographics
total_count = df['Demographics'].count()

# Calculate the total count for filtered groups
filtered_count = filtered_df['Demographics'].count()

print("Total count for Woman_Higer_degree:", filtered_count)
print("Total count of all Demographics:", total_count)


# In[61]:


#Assuming, WOMAN is a magor group for Degree, find distribution,%
total_women_higher_degree = 20715
total_professionals =27779
#percentage of women with higher degrees
percentage_women_higher_degree = (total_women_higher_degree / total_professionals) * 100
# Print the percentage with two decimal places
print(f"Percentage of Women with Higher Degrees: {percentage_women_higher_degree:.2f}%")


# In[62]:


#clean and standartise string
df['Age'] = df['Age'].str.upper()
df['Age'] = df['Age'].str.strip()


# In[63]:


df['Age'].unique()


# In[64]:


# Mapping old age groups to new ones
age_group_mapping = {
    '18-24': '18-25',
    '25-34': '25-35',
    '35-44': '35-50',
    '45-54': '35-50',  # Merging 35-44 and 45-54 into 35-50
    '55-64': '50-65',
    '65 OR OVER': '65+',
    'UNDER 18': '18'  # Keep 'UNDER 18' as it is
}

# Replace old age group labels with new ones
df['Age'] = df['Age'].replace(age_group_mapping)

# Show the updated DataFrame
df['Age'].unique()


# In[65]:


#create a new value 
df['Age_group'] = df['Age'] + '_' + df['Gender']


# In[66]:


# Sort df'Total_Income' in ascending order
df_sorted = df.sort_values(by='Total_Income', ascending=False)
df_sorted[['Total_Income', 'Currency', 'Job title', 'Highest_Degree','Age_group']].head(5)



# #clean data for range columns ‘Age’ , ‘Work Exp’ , ‘ Rel_exp’ :extract text, find unique groups /

# In[67]:


# identifying columns with missing data in updated df
print(df.isnull().sum())


# In[68]:


# identifying  number of unoque  values in 'Industry'
unique_industries = df['Industry'].nunique()
print("Number of unique industries:", unique_industries)


# In[69]:


# identifying  categories/values in a'Work_Exp' column
df['Work_Exp'].unique()


# In[70]:


# Frequency table for 'country   updated
Country_distribution = df['Country'].value_counts()

print(Country_distribution)


# After filtering unique values in a 'Country' string, Identifying missliding data and manually replase it :extract Country name or consider it missing, and  maark as 'Unknown'  using   .replace({'old_value1': 'new_value1'}).Using replace method clean 'Country'   from 

# In[71]:


#Remove parentheses using regular expression and replace method 
df['Country'] = df['Country'].str.replace(r'\(|\)', '', regex=True)


# In[72]:


#using reolace method , clean names form '.'
df['Country'] = df['Country'].str.replace('\.', '', regex=True)


# In[73]:


# make   all upper 
df['Country'] = df['Country'].str.upper()


# In[74]:


df['Country'] = df['Country'].str.strip()


# In[75]:


df['Country'].unique()


# In[76]:


# Replace the emoji with 'USA'
df['Country'] = df['Country'].str.replace('🇺🇸', 'USA')


# In[77]:


df['Country'] = df['Country'].str.replace('UNITED STATES I WORK FROM HOME AND MY CLIENTS ARE ALL OVER THE US/CANADA/PR', 'USA')


# In[78]:


df['Country'] = df['Country'].str.replace(r"I AM LOCATED IN CANADA BUT I WORK FOR A COMPANY IN THE US","USA", regex=True)


# In[79]:


# Replace values with 'USA'
df['Country'] = df['Country'].replace({'US GOVT EMPLOYEE OVERSEAS, COUNTRY WITHHELD': 'USA',
                                        'FOR THE UNITED STATES GOVERNMENT, BUT POSTED OVERSEAS': 'USA',
                                        'USA TOMORROW': 'USA'})


# In[80]:


df['Country'] = df['Country'].str.replace('USA COMPANY IS BASED IN A US TERRITORY, I WORK REMOTE', 'USA')


# In[81]:


import re
# Regular expression to match variations of USA and United States  use $   to mark end ($) of the string, 
usa_pattern = re.compile(r"^(USA|US|United States)$", flags=re.IGNORECASE)
# Replace matches with "USA", case-insensitive
df['Country'] = df['Country'].str.replace(usa_pattern, 'USA', regex=True)


# I naoticed multiply USA entityes to standartise it , i use strip()    and upper()   functions 

# I found a  464 group with name USA OF AMERICA'  ,I replase it with USA, but use  case sensitive function  case=False)

# In[82]:


df['Country'] = df['Country'].str.replace('UNITED STATES OF AMERICA ','USA')


# In[83]:


df['Country'] = df['Country'].str.replace('UNITEF STATED','USA')


# In[84]:


df['Country'] = df['Country'].str.replace('UNITED STATES OF AMERICA', 'USA', case=False)


# In[85]:


# Frequency table for 'country   updated
Country_distribution = df['Country'].value_counts()

print(Country_distribution)


# In[86]:


# Given value for USA occurrences
usa_total = 22956  # Already known

# Calculate the total number of occurrences of 'Total_Income' in the entire dataset
total_occurrences = df['Total_Income'].count()

# Calculate the percentage of USA's occurrences out of the total
percentage_usa_total_income = (usa_total / total_occurrences) * 100

# Print the results
print(f"Total occurrences of 'Total_Income': {total_occurrences}")

print(f"Percentage of USA's in 'Salary distribution': {percentage_usa_total_income:.2f}%")


# In[87]:


# Determine the distribution of categorical variables (e.g., Industry)
# Frequency table for 'Industry'
industry_distribution = df['Industry'].value_counts()

print(industry_distribution)



# In[88]:


# Frequency table for 'Job title'
Job_distribution = df['Job title'].value_counts()

print(Job_distribution)


# In[89]:


# identifying  categories/values in a targeted  variable 'Annual Salary' column
df['Job title'].unique()


# In[90]:


#  clean and standartise string
df['Job title'] = df['Job title'].str.upper()
df['Job title'] = df['Job title'].str.strip()


# In[91]:


# Identify duplicate rows where  in 'Job title' is 'Student' as most of responders - professionals 
duplicate_rows_df = df[df.duplicated(subset=['Job title'], keep=False) & (df['Job title'] == 'STUDENT')]

# Drop duplicate rows
df = df.drop(duplicate_rows_df.index)


# In[92]:


#Find a number of Jobs related to Software , using keywords 
keywords = ['Software', 'Lead',  'Senior Software Engineer', 'Program Manager']

# Filter the DataFrame based on keywords in the 'Job Title' column
Job_filtered_df = df[df['Job title'].str.contains('|'.join(keywords), case=False)]

# Print the filtered DataFrame
print(Job_filtered_df['Job title'])


# In[93]:


sorted_df = Job_filtered_df.sort_values(by='Total_Income', ascending=False)

# Print the desired columns from the sorted copy
sorted_df[['Total_Income', 'Age', 'Demographics', 'Currency', 'Job title', 'Rel_Exp']].head()


# In[94]:


from datetime import datetime 


# In[95]:


# Getting current date and time
current_time = datetime.now()

# Formatting the date and time in a readable format:
formatted_time = current_time.strftime('%B %d, %Y, %H:%M:%S')


# In[96]:


# Print the formatted date and time
print(f"Salary_survey_0.ipynb was last run on: {formatted_time}")


# NEW DF for   USA  country ,find outliers  .For targeted category 'Annial salary'=  find MEan  and median

# In[97]:


# Filter for US data
USA_filtered_df = df[df['Country'] == 'USA']

# Further filter for USD currency
USA_df_USD = USA_filtered_df[USA_filtered_df['Currency'] == 'USD']

# Sort by Total_Income in descending order
USA_df_USD_sorted = USA_df_USD.sort_values(by='Total_Income', ascending=False)

# Print the sorted DataFrame;''
USA_df_USD_sorted[['Total_Income', 'Age_group', 'Country','Job title', 'Rel_Exp','Highest_Degree']]


# Calculating   

# In[119]:


# Calculate IQR and identify outliers
Q1 = filtered_df['Annual Salary'].quantile(0.25)
Q3 = filtered_df['Annual Salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = filtered_df[(filtered_df['Annual Salary'] < Q1 - 1.5 * IQR) | (filtered_df['Annual Salary'] > Q3 + 1.5 * IQR)]

# Remove outliers (adjust as needed)
filtered_df_cleaned = filtered_df.drop(outliers.index)

# Calculate mean and median for cleaned data
mean_salary = filtered_df_cleaned['Annual Salary'].mean()
median_salary = filtered_df_cleaned['Annual Salary'].median()
print("Average Annual Salary for USA ,(USD): {:.2f}".format(mean_salary))
print("Mean Annual Salary for USA, (USD): {:.2f}".format(median_salary))


# In[99]:


# Find top salary perfomance for  USA
filtered_df = df[df['Country'] == 'USA']

# Sort by 'Annual Salary' in descending order
sorted_df = filtered_df.sort_values(by='Total_Income', ascending=False)

# Select the top 'n' rows (adjust 'n' as needed)
top_USA_salaries = sorted_df.head(n=10)  # For example, top 10 salaries

# Print the desired columns
top_USA_salaries[['Total_Income', 'State', 'Industry', 'Demographics']]


# In[100]:


filtered_USA_df = sorted_df[sorted_df['Country'] == 'USA']

# Calculate average salary by age group
average_salary = filtered_USA_df.groupby('Age_group')['Total_Income'].mean()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(average_salary.index, average_salary.values, color='coral')

# Adding labels and title
plt.xlabel('Average Annual Income ($)', fontsize=14)
plt.ylabel('Age Group', fontsize=14)
plt.title('Average income by Age Group in the USA ', fontsize=16)

# Adding average salary labels on top of each bar (adjusted x-offset)
for index, value in enumerate(average_salary.values):
    plt.text(value + 5000, index, f"${value:,.0f}", va='center', ha='right', fontsize=12)

plt.tight_layout()
plt.show()


# For  visual representation (poster)   create a bar chat only for AGe/USA /USD as  I did not apply   currensy convertation methods ,so use unfiltered DF will represent misleading data .

# In[101]:


filtered_USA_df = sorted_df[sorted_df['Country'] == 'USA']

# Calculate average salary by age group
average_salary = filtered_USA_df.groupby('Age')['Total_Income'].mean()

# Plotting the bar chart
plt.figure(figsize=(6, 4))
plt.barh(average_salary.index, average_salary.values, color='orange')

# Adding labels and title
plt.xlabel('Average Total Income ($)', fontsize=12,fontweight='bold')
plt.ylabel('Age Group', fontsize=12,fontweight='bold')
plt.title('Average Total Income by Age Group in the USA ', fontsize=12,fontweight='bold')

# Adding average salary labels on top of each bar (adjusted x-offset)
for index, value in enumerate(average_salary.values):
    plt.text(value + 5000, index, f"${value:,.0f}", va='center', ha='right', fontsize=12,fontweight='bold')

plt.tight_layout()
plt.show()


# In[125]:


# Create bar chart for  average salary /education in the filtered df (USA )
filtered_USA_df = sorted_df[sorted_df['Country'] == 'USA']
# Calculate
average_salary = filtered_USA_df.groupby('Highest_Degree')['Total_Income'].mean()
# Extract unique education levels
education_levels = average_salary.index.to_numpy()  

bar_width = 0.4 
# Plot bars for average salary
plt.bar(education_levels, average_salary, width=0.4, label='Average Salary', color='darkblue')

# Set labels and title
plt.xticks(education_levels, rotation=45, ha='right', fontweight='bold')
plt.xlabel('Education Level',fontweight='bold')
plt.ylabel('Average Salary (USD)',fontweight='bold')  
plt.title('Average Salary by Education Level (USA)',fontweight='bold')  #  title

# Set y-axis limits slightly higher than the maximum salary
plt.ylim(min(average_salary) - 3000, max(average_salary) + 800)

# Display the plot with grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




