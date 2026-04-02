import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
pf = pd.read_pickle('/users/sgdbareh/scratch/EPO_Experiments/EPO_Decision_Data/Train&TestData_2.0_PatentRefusal')

# Identify inventive step cases
inventive_step_mask = pf['Matched Articles'].astype(str).str.contains('Inventive Step', case=False, na=False)
inventive_step_cases = pf[inventive_step_mask]

# Filter for 2015-2024 and create outcome visualization
inventive_step_2015_2024 = inventive_step_cases[(inventive_step_cases['Year'] >= 2015) & (inventive_step_cases['Year'] <= 2024)]

# Create a cross-tabulation of Year vs Outcome
outcome_by_year = pd.crosstab(inventive_step_2015_2024['Year'], inventive_step_2015_2024['Outcome'])

# Create the bar plot
plt.figure(figsize=(14, 8))
outcome_by_year.plot(kind='bar', stacked=False, width=0.8)
plt.title('Outcome of Inventive Step Cases by Year (2015-2024)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Cases', fontsize=12)
plt.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('/users/sgdbareh/scratch/EPO_Experiments/EPO_data_stuff/inventive_step_outcomes_2015_2024.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("="*60)
print("INVENTIVE STEP CASES OUTCOME SUMMARY (2015-2024)")
print("="*60)
print("\nOutcome counts by year:")
print(outcome_by_year)

print("\nOutcome percentages by year:")
outcome_percentages = outcome_by_year.div(outcome_by_year.sum(axis=1), axis=0) * 100
print(outcome_percentages.round(2))

print(f"\nTotal inventive step cases 2015-2024: {len(inventive_step_2015_2024)}")
print(f"Most common outcome: {inventive_step_2015_2024['Outcome'].mode().iloc[0]}")
print(f"Outcome distribution:")
print(inventive_step_2015_2024['Outcome'].value_counts())

# Sample 20 cases: 10 affirmed and 10 reversed
print("\n" + "="*60)
print("SAMPLING 20 INVENTIVE STEP CASES")
print("="*60)

# Set random seed for reproducibility
np.random.seed(42)

# Filter out 'Unknown' outcomes for sampling
inventive_step_clean = inventive_step_cases[inventive_step_cases['Outcome'].isin(['Affirmed', 'Reversed'])]

# Sample 10 affirmed cases
affirmed_cases = inventive_step_clean[inventive_step_clean['Outcome'] == 'Affirmed']
if len(affirmed_cases) >= 10:
    sampled_affirmed = affirmed_cases.sample(n=10, random_state=42)
else:
    sampled_affirmed = affirmed_cases

# Sample 10 reversed cases
reversed_cases = inventive_step_clean[inventive_step_clean['Outcome'] == 'Reversed']
if len(reversed_cases) >= 10:
    sampled_reversed = reversed_cases.sample(n=10, random_state=42)
else:
    sampled_reversed = reversed_cases

# Combine the samples
sampled_cases = pd.concat([sampled_affirmed, sampled_reversed], ignore_index=True)

# Create the output text
output_text = "SAMPLE OF 20 INVENTIVE STEP CASES\n"
output_text += "="*50 + "\n\n"
output_text += f"Random seed: 42\n"
output_text += f"Total cases  sampled: {len(sampled_cases)}\n"
output_text += f"Affirmed cases: {len(sampled_affirmed)}\n"
output_text += f"Reversed cases: {len(sampled_reversed)}\n\n"

output_text += "AFFIRMED CASES:\n"
output_text += "-" * 20 + "\n"
for i, (idx, case) in enumerate(sampled_affirmed.iterrows(), 1):
    output_text += f"{i}. Reference: {case['Reference']}\n"
    output_text += f"   Year: {case['Year']}\n"
    output_text += f"   Outcome: {case['Outcome']}\n"
    output_text += f"   Matched Articles: {case['Matched Articles']}\n"
    output_text += f"   Invention Title: {case['Invention Title']}\n\n"

output_text += "REVERSED CASES:\n"
output_text += "-" * 20 + "\n"
for i, (idx, case) in enumerate(sampled_reversed.iterrows(), 1):
    output_text += f"{i}. Reference: {case['Reference']}\n"
    output_text += f"   Year: {case['Year']}\n"
    output_text += f"   Outcome: {case['Outcome']}\n"
    output_text += f"   Matched Articles: {case['Matched Articles']}\n"
    output_text += f"   Invention Title: {case['Invention Title']}\n\n"

# Save to text file
output_file = '/users/sgdbareh/scratch/EPO_Experiments/EPO_data_stuff/sampled_inventive_step_cases.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(output_text)

print(f"Sampled cases saved to: {output_file}")
print(f"Affirmed cases sampled: {len(sampled_affirmed)}")
print(f"Reversed cases sampled: {len(sampled_reversed)}")

# Display the references for quick reference
print("\nSampled case references:")
print("Affirmed cases:")
for ref in sampled_affirmed['Reference'].tolist():
    print(f"  - {ref}")
print("Reversed cases:")
for ref in sampled_reversed['Reference'].tolist():
    print(f"  - {ref}")

