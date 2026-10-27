# Universal AI Data Analyst: 20 Viva Voce Questions & Answers

Defend your final year project with confidence using these comprehensive technical explanations.

---

### Q1: What makes this system "Universal" rather than hardcoded to a specific dataset?
**Answer**:
Our ingestion engine ([src/loader.py](file:///c:/Users/Hari/Desktop/Final%20Year/src/loader.py)) dynamically inspects uploaded files at runtime. It utilizes byte sniffing for encoding detection (`utf-8`, `latin-1`), `csv.Sniffer` for delimiter inference, recursive AST type checks for schema inference, and dynamic column transformers in scikit-learn. No column name or data type is hardcoded anywhere in the codebase.

---

### Q2: How does the system prevent Data Leakage during automated preprocessing?
**Answer**:
Data leakage is prevented in two critical ways:
1. **Partition-level Leakage**: We calculate MD5 checksums of individual rows and images across train, validation, and test splits to ensure zero identical records exist across partitions.
2. **Preprocessing-level Leakage**: We employ `sklearn.pipeline.Pipeline` combined with `ColumnTransformer`. Parameters such as median imputation values and standard deviation scalers are computed **strictly on the training partition** (`fit`) and only mapped to validation/test partitions (`transform`).

---

### Q3: Why did you choose Interquartile Range (IQR) over Z-Score for outlier handling?
**Answer**:
The Z-Score assumes a Gaussian (normal) distribution and is vulnerable to extreme outliers because the mean and standard deviation themselves are sensitive to outliers. IQR ($Q_3 - Q_1$) is non-parametric and resistant to skewness, making it far more robust for universal, unknown datasets.

---

### Q4: How does the AutoML engine distinguish between Classification and Regression?
**Answer**:
The system analyzes target column data types and cardinality:
- If the target is categorical, boolean, or string $\rightarrow$ **Classification**.
- If the target is numeric with $\le 10$ unique discrete values $\rightarrow$ **Classification**.
- If the target is numeric with $> 10$ unique continuous values $\rightarrow$ **Regression**.

---

### Q5: How does the Ask-Your-Data Chatbot protect user data privacy?
**Answer**:
We never transmit the full dataset to the LLM API. Instead, we generate a compact schema summary containing only column names, data types, and 3-5 sanitized sample rows. The LLM acts purely as a code synthesizer, returning pandas instructions that execute locally on the user's machine.

---

### Q6: How do you prevent malicious code injection in the LLM's generated Python code?
**Answer**:
In [src/chatbot.py](file:///c:/Users/Hari/Desktop/Final%20Year/src/chatbot.py), we implement a multi-layered security sandbox:
1. **Keyword Filtering**: Rejects strings containing `os`, `sys`, `subprocess`, `eval`, `open`, `__import__`, socket, etc.
2. **AST (Abstract Syntax Tree) Verification**: Traverses syntax nodes to prohibit unauthorized imports.
3. **Restricted Execution Scope**: Executes within an isolated namespace where `__builtins__` is constrained and only `pd`, `np`, and `px` are injected.

---

### Q7: How does the Voice & Tamil translation pipeline function?
**Answer**:
1. Spoken audio from the user's browser microphone is sent to Groq's `whisper-large-v3-turbo` model for sub-second speech-to-text.
2. If the prompt is in Tamil, Groq translates the analytical intent into English pandas manipulation code while keeping exact column names untouched.
3. The query executes locally on the DataFrame.
4. The synthesized answer is read aloud using `gTTS` with the `ta` (Tamil) or `en` (English) language synthesizer.

---

### Q8: What if the Groq API key is expired or the user is offline during the viva demo?
**Answer**:
We built an offline, deterministic rule-based analyst engine ([src/chatbot.py](file:///c:/Users/Hari/Desktop/Final%20Year/src/chatbot.py) `offline_fallback_analyst`) that parses common analytical queries (row counts, column lists, missing values, descriptive statistics, top values) using direct pandas operations so the chatbot never crashes.

---

### Q9: Why is MobileNetV2 used for computer vision rather than training a CNN from scratch?
**Answer**:
MobileNetV2 leverages depthwise separable convolutions, dramatically reducing parameter count (~3.4M parameters) while maintaining high representational capacity. Transfer learning with ImageNet pre-trained weights allows the model to converge rapidly even on small student datasets without overfitting.

---

### Q10: How does the system handle high-cardinality identifier columns?
**Answer**:
Columns matching regex patterns for IDs (e.g. `PassengerId`, `OrderID`, names) or columns where every single row has a unique value ($nunique == N$) are automatically dropped before model fitting to prevent artificial memorization and overfitting.

---

### Q11: What is the purpose of Cross-Validation in the AutoML benchmark?
**Answer**:
Evaluating models solely on a single split can lead to variance based on lucky or unlucky sampling. We apply 3-fold cross-validation on the training set to evaluate stability before reporting the final held-out test score.

---

### Q12: Why did you redesign the user interface into Neobrutalism?
**Answer**:
Traditional Streamlit dashboards rely on cluttered, dated sidebar navigation. Modern web standards favor high-contrast Neobrutalism (bold `2.5px` borders, offset hard shadows, floating capsule top-bars, and status pills), improving visual hierarchy and user task completion speed.

---

### Q13: How does the system handle imbalanced classification datasets?
**Answer**:
1. We compute class frequency ratios and trigger an imbalance warning if the majority-to-minority ratio exceeds $3:1$.
2. We enforce `stratified` sampling during train/validation/test splitting to preserve equal class proportions.
3. We report weighted F1-score rather than simple accuracy to prevent deceptive performance inflation.

---

### Q14: How are multi-sheet Excel files handled?
**Answer**:
The loader uses `pd.ExcelFile` to inspect the workbook index and presents an interactive sheet selector dropdown, allowing users to choose the target sheet dynamically before ingestion.

---

### Q15: How does the automated report generator compile documents without external Office software?
**Answer**:
- For PDF: We utilize `reportlab`, a pure-Python canvas and flowable document generator.
- For Word: We utilize `python-docx`, which manipulates OpenXML packages directly in memory to produce compliant `.docx` files.

---

### Q16: What happens if an uploaded dataset exceeds memory limits?
**Answer**:
Files larger than 50MB trigger an automated banner alerting the user and offering deterministic row downsampling (e.g., first 50,000 rows) to ensure browser responsiveness.

---

### Q17: Why use Plotly instead of Matplotlib/Seaborn?
**Answer**:
Plotly renders interactive WebGL/SVG graphics directly in the browser with hover tooltips, zoom, box-select, and responsive scaling, offering a superior UX compared to static bitmap images.

---

### Q18: What is the difference between Winsorization (clipping) and Outlier Dropping?
**Answer**:
Outlier dropping purges the entire row, which can discard valuable records in small datasets. Winsorization (clipping) caps extreme values to the $Q_1 - 1.5 \times IQR$ and $Q_3 + 1.5 \times IQR$ thresholds, preserving the sample size while stabilizing variance.

---

### Q19: How are date and time trends analyzed if the date format is ambiguous?
**Answer**:
`src/cleaner.py` evaluates sample values with `pd.to_datetime` using fallback parser cascades. Once converted to `datetime64[ns]`, the data is chronologically sorted and aggregated using time-windowed resamplers.

---

### Q20: How can this system be extended for production deployment?
**Answer**:
The modular separation of `src/` makes it trivial to wrap the loaders, cleaners, and AutoML engines into FastAPI endpoints, containerize with Docker, and deploy to Kubernetes or AWS ECS with GPU worker pools.
