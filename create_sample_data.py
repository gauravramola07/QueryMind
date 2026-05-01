# create_sample_data.py

import pandas as pd
import numpy as np

# Create a realistic sales dataset
np.random.seed(42)
n = 1000

data = {
    'order_id': range(1001, 1001 + n),
    'date': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%Y-%m-%d'),
    'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'], n),
    'category': np.random.choice(['Electronics', 'Accessories', 'Wearables'], n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'sales_rep': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], n),
    'quantity': np.random.randint(1, 20, n),
    'unit_price': np.random.choice([299, 599, 999, 1299, 199], n),
    'discount': np.random.choice([0, 5, 10, 15, 20], n),
    'revenue': np.random.randint(500, 50000, n),
    'profit': np.random.randint(100, 15000, n),
    'customer_rating': np.round(np.random.uniform(3.0, 5.0, n), 1)
}

df = pd.DataFrame(data)
df.to_csv('data/sample_sales_data.csv', index=False)

print('✅ Sample dataset created!')
print(f'Shape: {df.shape}')
print(df.head())