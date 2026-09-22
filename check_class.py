import pandas as pd
df = pd.read_csv('results/ensemble_N32/ensemble.csv')
sub = df[(df.fi_level == 1) & (df.bit_class == 'mant_hi')]
print(sub['class'].value_counts())
