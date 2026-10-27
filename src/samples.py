"""
=============================================================================
Sample Dataset Generator for Instant Testing & Viva Demonstration (src/samples.py)
=============================================================================
Generates ready-to-test real datasets:
1. Titanic Survival (CSV) - Binary Classification with missing values & mixed types
2. Iris Flowers (CSV) - Multi-class classification
3. E-Commerce Sales (CSV) - Regression & Time-Series trend analysis
4. Multi-class Image Folder & ZIP (cats/ vs dogs/) - Computer vision transfer learning
"""

import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "samples"))


def ensure_sample_datasets() -> dict:
    """Creates sample CSV datasets and image classification zip if not already present."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    paths = {}

    # 1. Titanic Dataset
    titanic_path = os.path.join(SAMPLES_DIR, "titanic.csv")
    if not os.path.exists(titanic_path):
        np.random.seed(42)
        n = 200
        pclass = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
        sex = np.random.choice(["male", "female"], size=n, p=[0.64, 0.36])
        age = np.random.normal(29, 14, size=n).round(1)
        age = np.clip(age, 1, 80)
        # Add 15% realistic missing values in age
        age_mask = np.random.rand(n) < 0.15
        age[age_mask] = np.nan
        
        fare = np.where(pclass == 1, np.random.exponential(80, n),
               np.where(pclass == 2, np.random.exponential(25, n),
                                     np.random.exponential(13, n))).round(2)
        embarked = np.random.choice(["S", "C", "Q", None], size=n, p=[0.7, 0.2, 0.08, 0.02])
        
        # Probabilistic survival depending on sex and class
        prob = np.where(sex == "female", 0.74, 0.19) - (pclass - 1) * 0.15
        prob = np.clip(prob, 0.05, 0.95)
        survived = (np.random.rand(n) < prob).astype(int)

        df_titanic = pd.DataFrame({
            "PassengerId": range(1, n + 1),
            "Survived": survived,
            "Pclass": pclass,
            "Name": [f"Passenger, Person {i}" for i in range(1, n + 1)],
            "Sex": sex,
            "Age": age,
            "SibSp": np.random.choice([0, 1, 2, 3], size=n, p=[0.68, 0.23, 0.06, 0.03]),
            "Parch": np.random.choice([0, 1, 2], size=n, p=[0.76, 0.16, 0.08]),
            "Fare": fare,
            "Embarked": embarked,
        })
        df_titanic.to_csv(titanic_path, index=False)
    paths["titanic"] = titanic_path

    # 2. E-Commerce Sales Dataset
    sales_path = os.path.join(SAMPLES_DIR, "sales_data.csv")
    if not os.path.exists(sales_path):
        np.random.seed(101)
        m = 300
        dates = pd.date_range(start="2024-01-01", periods=m, freq="D")
        categories = np.random.choice(["Electronics", "Clothing", "Home & Kitchen", "Books"], size=m)
        regions = np.random.choice(["North", "South", "East", "West"], size=m)
        units = np.random.randint(1, 15, size=m)
        unit_prices = {
            "Electronics": 350.0,
            "Clothing": 45.0,
            "Home & Kitchen": 75.0,
            "Books": 20.0,
        }
        base_prices = np.array([unit_prices[c] for c in categories])
        revenue = (units * base_prices * np.random.uniform(0.85, 1.15, size=m)).round(2)
        profit = (revenue * np.random.uniform(0.12, 0.35, size=m)).round(2)
        
        df_sales = pd.DataFrame({
            "OrderID": [f"ORD-{1000+i}" for i in range(m)],
            "OrderDate": dates.strftime("%Y-%m-%d"),
            "Category": categories,
            "Region": regions,
            "UnitsSold": units,
            "Revenue": revenue,
            "Profit": profit,
        })
        df_sales.to_csv(sales_path, index=False)
    paths["sales"] = sales_path

    # 3. Sample Image Classification Folder & ZIP (cats vs dogs)
    img_root = os.path.join(SAMPLES_DIR, "animal_images")
    zip_path = os.path.join(SAMPLES_DIR, "animal_images.zip")
    
    if not os.path.exists(zip_path):
        os.makedirs(os.path.join(img_root, "cats"), exist_ok=True)
        os.makedirs(os.path.join(img_root, "dogs"), exist_ok=True)

        # Generate 20 distinct synthetic images per class with identifiable visual patterns
        for i in range(20):
            # Cat image: warm orange/gold circles & triangular ears pattern
            cat_img = Image.new("RGB", (128, 128), color=(255, 235, 204))
            draw = ImageDraw.Draw(cat_img)
            draw.ellipse([20 + (i % 5)*5, 20 + (i % 5)*5, 100, 100], fill=(230, 115, 0))
            draw.polygon([(30, 20), (50, 50), (10, 50)], fill=(200, 80, 0))
            draw.polygon([(90, 20), (110, 50), (70, 50)], fill=(200, 80, 0))
            cat_img.save(os.path.join(img_root, "cats", f"cat_{i+1:02d}.jpg"))

            # Dog image: cool blue/brown rectangular patches & snout pattern
            dog_img = Image.new("RGB", (128, 128), color=(210, 230, 250))
            draw = ImageDraw.Draw(dog_img)
            draw.rectangle([25 + (i % 4)*6, 30, 95, 95], fill=(70, 130, 180))
            draw.ellipse([45, 60, 75, 85], fill=(40, 40, 40))
            dog_img.save(os.path.join(img_root, "dogs", f"dog_{i+1:02d}.jpg"))

        # Create zip archive
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(img_root):
                for file in files:
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, img_root)
                    zipf.write(full_p, rel_p)

    paths["image_folder"] = img_root
    paths["image_zip"] = zip_path

    return paths
