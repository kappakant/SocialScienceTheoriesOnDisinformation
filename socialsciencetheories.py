import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import re
import math
import textstat
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, Counter

# NLP and ML
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import xgboost as xgb
from xgboost import XGBClassifier

# Deep Learning
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)
from tqdm import tqdm
import shap

warnings.filterwarnings("ignore")

# GossipCop in this example
training = pd.read_csv('~/csvs/gossipcop_train.csv', index_col=0).dropna()
testing = pd.read_csv('~/csvs/gossipcop_test.csv', index_col=0).dropna()

training['text'] = training['text'].str.lower()
testing['text'] = testing['text'].str.lower()

gossipcop_fake = pd.read_csv('~/csvs/gossipcop_fake.csv').dropna()
gossipcop_real = pd.read_csv('~/csvs/gossipcop_real.csv').dropna()

fake_urls = gossipcop_fake[['id', 'news_url']].copy()
real_urls = gossipcop_real[['id', 'news_url']].copy()
fake_urls['id'] = fake_urls['id'].astype(str)
real_urls['id'] = real_urls['id'].astype(str)

id2url = (
    pd.concat([fake_urls, real_urls], ignore_index=True)
      .dropna(subset=['id', 'news_url'])
      .drop_duplicates(subset=['id'], keep='first')
      .rename(columns={'news_url': 'url_from_id'})
)

def ensure_id_column(df):
    if 'id' in df.columns:
        out = df.copy()
        out['id'] = out['id'].astype(str)
        return out
    else:
        out = df.reset_index().rename(columns={'index': 'id'}).copy()
        out['id'] = out['id'].astype(str)
        return out

training_wid = ensure_id_column(training)
testing_wid = ensure_id_column(testing)

training_merged = training_wid.merge(id2url, on='id', how='left')
testing_merged = testing_wid.merge(id2url, on='id', how='left')

training_merged.rename(columns={'url_from_id': 'url'}, inplace=True)
testing_merged.rename(columns={'url_from_id': 'url'}, inplace=True)

matched_train = training_merged['url'].notna().sum()
matched_test = testing_merged['url'].notna().sum()
print(f"[training] matched URLs: {matched_train} / {len(training_merged)}")
print(f"[testing] matched URLs: {matched_test} / {len(testing_merged)}")

train_data = training_merged
test_data = testing_merged

X_train = train_data['text']
y_train = train_data['label']
X_test = test_data['text']
y_test = test_data['label']

training = training_merged
testing = testing_merged

# Theory 5 feature: Internal Validation (Logical Consistency)
def detect_logical_consistency(texts):
    entailment_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=0)
    
    consistency_scores = []
    for text in tqdm(texts, desc="Processing Logical Consistency", unit="text"):
        sentences = text.split('. ')  
        if len(sentences) < 2:
            consistency_scores.append(1.0)  
            continue

        consistent_count = 0
        total_pairs = 0
        for i in range(len(sentences) - 1):
            premise = sentences[i]
            hypothesis = sentences[i + 1]
            if len(premise) == 0 or len(hypothesis) == 0:
                continue

            result = entailment_pipeline(
                f"{premise} [SEP] {hypothesis}",
                truncation=True
            )
            if result[0]['label'] == 'ENTAILMENT':
                consistent_count += 1
            total_pairs += 1
        
        if total_pairs == 0:
            consistency_scores.append(1.0)  
        else:
            score = consistent_count / total_pairs 
            consistency_scores.append(score)

    return np.array(consistency_scores, dtype=np.float32)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('theory5_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return accuracy

print("Extracting logical consistency features for training set...")
X_train_consistency = detect_logical_consistency(X_train)
print("Extracting logical consistency features for testing set...")
X_test_consistency = detect_logical_consistency(X_test)

accuracy = train_and_evaluate(X_train_consistency, X_test_consistency, y_train, y_test)
print(f'Model Accuracy: {accuracy:.4f}')

# Theory 6 feature: Coherence
nlp = spacy.load('en_core_web_sm')

def get_entity_grid(doc):
    entity_grid = defaultdict(list)
    for sent in doc.sents:
        for token in sent:
            if token.ent_type_ or token.pos_ in ("PROPN", "NOUN"):
                if token.dep_ in ("nsubj","nsubjpass"):
                    role = "S"
                elif token.dep_ in ("dobj","obj","pobj"):
                    role = "O"
                else:
                    role = "X"
                entity_grid[token.lemma_].append(role)
    return entity_grid

def compute_entity_grid_coherence(text):
    doc = nlp(text)
    entity_grid = get_entity_grid(doc)

    all_transitions = []
    for roles in entity_grid.values():
        for i in range(len(roles)-1):
            all_transitions.append((roles[i], roles[i+1]))

    if not all_transitions:
        return None

    trans_counts = Counter(all_transitions)
    total = sum(trans_counts.values())
    probabilities = [count / total for count in trans_counts.values()]

    coherence_score = sum(math.log(p) for p in probabilities) / len(probabilities)
    return coherence_score

testing['coherence_score'] = testing['text'].apply(compute_entity_grid_coherence)
training['coherence_score'] = training['text'].apply(compute_entity_grid_coherence)

# Theory 7 features: Credibility Source
le = LabelEncoder()
training['publisher_encoded'] = le.fit_transform(training['url'].astype(str))
testing['publisher_encoded'] = le.fit_transform(testing['url'].astype(str))

# Theory 8 feature: Supporting Evidence
ATTRIB_PAT = re.compile(r"\b(according to|reported by|said|stated|told|a report by|as per|as reported by)\b", re.I)
URL_PAT = re.compile(r'https?://\S+|www\.\S+', re.I)
QUOTE_PAT = re.compile(r'["""\'''].+?["""\''']')
DATE_PAT = re.compile(r'\b(20\d{2}|19\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b', re.I)

TRUST_HINTS = (
    ".gov", ".edu", "/doi/", "arxiv.org", "nature.com", "science.org",
    "who.int", "cdc.gov", "un.org", "oecd.org", "imf.org"
)

def supporting_evidence_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0

    t = text[:10000]
    n_tokens = max(1, len(t.split()))
    k = 1000.0 / n_tokens

    urls = URL_PAT.findall(t)
    n_urls = len(urls)
    n_quotes = len(QUOTE_PAT.findall(t))
    n_dates = len(DATE_PAT.findall(t))
    n_attrib = len(ATTRIB_PAT.findall(t))

    trust_ratio = 0.0
    if n_urls > 0:
        trust_urls = sum(any(h in u.lower() for h in TRUST_HINTS) for u in urls)
        trust_ratio = trust_urls / n_urls

    f_urls = min(1.0, (n_urls * k) / 1.0)
    f_quotes = min(1.0, (n_quotes * k) / 2.0)
    f_dates = min(1.0, (n_dates * k) / 2.0)
    f_attrib = min(1.0, (n_attrib * k) / 2.0)
    f_trust = trust_ratio

    feats = [f_urls, f_quotes, f_dates, f_attrib, f_trust]
    score = sum(feats) / 5.0

    return float(max(0.0, min(1.0, score)))

training["supporting_evidence"] = training["text"].map(supporting_evidence_score)
testing["supporting_evidence"] = testing["text"].map(supporting_evidence_score)

# %% Theory 9 features: Language Vagueness
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv('~/csvs/vague_phrases_scores.csv')

assert {'vague_phrases','scores'}.issubset(df.columns), "unmatched, check your column names"

df['score_01'] = (df['scores'] - df['scores'].min()) / (df['scores'].max() - df['scores'].min())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df[['vague_phrases', 'score_01']].rename(columns={'vague_phrases':'text','score_01':'labels'}))
test_dataset = Dataset.from_pandas(test_df[['vague_phrases', 'score_01']].rename(columns={'vague_phrases':'text','score_01':'labels'}))

Vague_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
Vague_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=1,
    problem_type="regression"
).to(device)

def preprocess(batch):
    enc = Vague_tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    enc['labels'] = batch['labels']
    return enc

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(preprocess, batched=True, remove_columns=test_dataset.column_names)

train_dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
test_dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

training_args = TrainingArguments(
    output_dir="./vagueness_model",
    eval_strategy="epoch",
    logging_strategy="epoch",  
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=Vague_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=Vague_tokenizer
)

print("Training BERT vagueness REGRESSION model (outputs in [0,1]) ...")
trainer.train()

best_dir = "./vagueness_model_best"
trainer.save_model(best_dir)
Vague_tokenizer.save_pretrained(best_dir)

Vague_tokenizer = BertTokenizer.from_pretrained(best_dir)
Vague_model = BertForSequenceClassification.from_pretrained(best_dir).to(device)
Vague_model.eval()

@torch.no_grad()
def get_vagueness_score(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    inputs = Vague_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = Vague_model(**inputs)
    val = outputs.logits.squeeze().item()  
    return max(0.0, min(1.0, val))

tqdm.pandas()
training['vagueness_score'] = training['text'].progress_apply(get_vagueness_score)
testing['vagueness_score'] = testing['text'].progress_apply(get_vagueness_score)

print("Vagueness completed successfully, cleaning cache now")
del Vague_model
del Vague_tokenizer
torch.cuda.empty_cache()
gc.collect()

# Theory 10 features: Information Gap
training_title = training.copy()
testing_title = testing.copy()

training_title['title'] = training_title['title'].str.lower()
testing_title['title'] = testing_title['title'].str.lower()

# Needs "eager" attn_implementation due to issue with our computer cluster, consider different attn_implementation for your case.
LLM_model_name = "google/gemma-3-27b-it"
LLM_tokenizer = AutoTokenizer.from_pretrained(LLM_model_name)
LLM_model = AutoModelForCausalLM.from_pretrained(
    LLM_model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
).eval()
device = next(LLM_model.parameters()).device

# Part 1: Punctuation
PUNCT_END_RE = re.compile(r"(\\.\\.\\.|[?!\\.])$")

def tail_punct(title: str) -> Optional[str]:
    if not isinstance(title, str) or not title:
        return None
    t = title.rstrip()
    m = PUNCT_END_RE.search(t)
    if not m:
        return None
    return m.group(1)

def _hf_first_id(tokenizer, tok: str) -> Optional[int]:
    ids = tokenizer.encode(tok, add_special_tokens=False)
    return ids[0] if ids else None

def _sequence_logprob(prefix_ids: torch.LongTensor, seq_token_ids: List[int], model: torch.nn.Module) -> float:
    device = next(model.parameters()).device
    cur = prefix_ids.to(device)
    total = 0.0
    with torch.no_grad():
        for tid in seq_token_ids:
            out = model(input_ids=cur)
            step_logprobs = torch.log_softmax(out.logits[0, -1, :], dim=-1)
            total += float(step_logprobs[tid].item())
            cur = torch.cat([cur, torch.tensor([[tid]], dtype=cur.dtype, device=device)], dim=1)
    return total

def predict_punct_probs(text_wo_trailing_punct: str) -> Dict[str, float]:
    enc = LLM_tokenizer(text_wo_trailing_punct, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = LLM_model(**enc)
        last_logits = out.logits[0, -1, :]
        last_logprobs = torch.log_softmax(last_logits, dim=-1)

    q_id = _hf_first_id(LLM_tokenizer, "?")
    ex_id = _hf_first_id(LLM_tokenizer, "!")
    dot_id = _hf_first_id(LLM_tokenizer, ".")
    ellipsis_ids = LLM_tokenizer.encode("...", add_special_tokens=False)

    logp_q = float(last_logprobs[q_id].item()) if q_id is not None else -1e9
    logp_ex = float(last_logprobs[ex_id].item()) if ex_id is not None else -1e9
    if len(ellipsis_ids) == 0:
        logp_ell = -1e9
    elif len(ellipsis_ids) == 1:
        logp_ell = float(last_logprobs[ellipsis_ids[0]].item())
    else:
        prefix_ids = enc["input_ids"]
        logp_ell = _sequence_logprob(prefix_ids, ellipsis_ids, LLM_model)
    logp_dot = float(last_logprobs[dot_id].item()) if dot_id is not None else -1e9

    xs = torch.tensor([logp_q, logp_ex, logp_ell, logp_dot], dtype=torch.float32)
    ps = torch.softmax(xs, dim=-1).tolist()
    return {"?": ps[0], "!": ps[1], "...": ps[2], ".": ps[3]}

def punctuation_score(title: str) -> float:
    tail = tail_punct(title)
    if tail in {"?", "...", "!"}:
        return 1.0
    if tail == ".":
        return 0.0
    stripped = title.rstrip(" .!?")
    probs = predict_punct_probs(stripped)
    return float(probs["?"] + probs["!"] + probs["..."])

# Part 2: Surprisingness
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def fit_lda_and_baseline(titles: List[str], n_topics: int = 20, max_features: int = 20000, random_state: int = 0):
    vectorizer = CountVectorizer(max_features=max_features, lowercase=True, stop_words='english')
    X = vectorizer.fit_transform(titles)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    doc_topic = lda.fit_transform(X)
    Q = np.maximum(doc_topic.mean(axis=0), 1e-8)
    Q = Q / Q.sum()
    return vectorizer, lda, Q

def infer_topic_distribution(title: str, vectorizer, lda_model) -> np.ndarray:
    X = vectorizer.transform([title])
    P = lda_model.transform(X)[0]
    P = np.maximum(P, 1e-12)
    P = P / P.sum()
    return P

def surprisingness_score(title: str, vectorizer, lda_model, Q: np.ndarray) -> Tuple[float, float]:
    P = infer_topic_distribution(title, vectorizer, lda_model)
    dkl = kl_divergence(P, Q)
    sscore = float(dkl / (1.0 + dkl))
    return sscore, dkl

# Train
print("Scoring training titles ...")
titles = training_title['title'].fillna("").astype(str).tolist()
vectorizer, lda, Q = fit_lda_and_baseline(titles, n_topics=20)

w_punct, w_surp = 0.5, 0.5 
punct_scores, surp_scores, kls = [], [], []

for title in tqdm(training_title['title'].fillna("").astype(str).tolist(), desc="Scoring titles"):
    pscore = punctuation_score(title)                  
    sscore, dkl = surprisingness_score(title, vectorizer, lda, Q)
    punct_scores.append(pscore)
    surp_scores.append(sscore)
    kls.append(dkl)

training_title['punctuation_score'] = punct_scores
training_title['surprisingness_score'] = surp_scores
training_title['KL'] = kls
training_title['information_gap_score'] = w_punct * training_title['punctuation_score'] + w_surp * training_title['surprisingness_score']

# Test
print("Scoring testing titles ...")
titles = testing_title['title'].fillna("").astype(str).tolist()
vectorizer, lda, Q = fit_lda_and_baseline(titles, n_topics=20)
w_punct, w_surp = 0.5, 0.5 
punct_scores, surp_scores, kls = [], [], []
for title in tqdm(testing_title['title'].fillna("").astype(str).tolist(), desc="Scoring titles"):
    pscore = punctuation_score(title)                   
    sscore, dkl = surprisingness_score(title, vectorizer, lda, Q)
    punct_scores.append(pscore)
    surp_scores.append(sscore)
    kls.append(dkl)
testing_title['punctuation_score'] = punct_scores
testing_title['surprisingness_score'] = surp_scores
testing_title['KL'] = kls
testing_title['information_gap_score'] = w_punct * testing_title['punctuation_score'] + w_surp * testing_title['surprisingness_score']

# Theory 1 Feature: Negativity Bias
analyzer = SentimentIntensityAnalyzer()

def get_compound(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

training['compound'] = training['text'].apply(get_compound)
testing['compound'] = testing['text'].apply(get_compound)

# Theory 2 Feature: Amplification Hypothesis
certain_words = {
    'absolute', 'absolutely', 'all', 'always', 'apparent', 'assured', 'certain', 'clear', 'clearly',
    'complete', 'confident', 'definitely', 'distinct', 'evident', 'fact', 'factual', 'forever',
    'fundamental', 'guaranteed', 'indeed', 'inevitable', 'obviously', 'positive', 'precise', 
    'proof', 'pure', 'sure', 'true', 'wholly'
}

uncertain_words = {
    'maybe', 'perhaps', 'possibly', 'doubt', 'doubtfully',
    'questionable', 'unsure', 'unproven', 'unknown', 'unclear'
}

def classify_certainty(text):
    if not isinstance(text, str):  
        return 0  

    text_lower = text.lower()

    certain_count = sum(len(re.findall(rf'\b{word}\b', text_lower)) for word in certain_words)
    uncertain_count = sum(len(re.findall(rf'\b{word}\b', text_lower)) for word in uncertain_words)

    if certain_count > uncertain_count:
        return 1
    else:
        return 0

training['amplification_label'] = training['text'].apply(classify_certainty)
testing['amplification_label'] = testing['text'].apply(classify_certainty)

# Theory 3 Feature: Scarcity Principle
def get_scarcity_label(text):
    truncated_text = text[:6000]
    
    prompt = f"""Scarcity bias means a message becomes more persuasive when it emphasizes limited availability, urgency, or time-sensitivity. 
For example: "Only today", "Hurry up before it's gone", or "This info may be deleted soon".

Determine whether the following news text uses scarcity bias tactics and respond with ONLY one number:
0 (does NOT use scarcity bias), 1 (uses scarcity bias).

News: {truncated_text}

Answer:"""

    input_ids = LLM_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = LLM_model.generate(**input_ids, max_new_tokens=10)
    response = LLM_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    score = None
    for token in reversed(response.split()):
        if token in ['0', '1']:
            score = token
            break
    
    return int(score)

print("Processing training data for scarcity bias...")
training["scarcity_label"] = [get_scarcity_label(text) for text in tqdm(training["text"])]

print("\nProcessing test data for scarcity bias...")
testing["scarcity_label"] = [get_scarcity_label(text) for text in tqdm(testing["text"])]

del LLM_model
del LLM_tokenizer
torch.cuda.empty_cache()
gc.collect()

# Theory 4 Feature: Information Complexity
def calculate_text_complexity(text):
    return textstat.flesch_reading_ease(text)

training['info_complexity'] = training['text'].apply(calculate_text_complexity)
testing['info_complexity'] = testing['text'].apply(calculate_text_complexity)

# FINAL FND
X_train = np.hstack((
    training["compound"].values.reshape(-1, 1),
    training["amplification_label"].values.reshape(-1, 1),
    training["scarcity_label"].values.reshape(-1, 1),
    training["info_complexity"].values.reshape(-1, 1),
    X_train_consistency.reshape(-1, 1),
    training["coherence_score"].values.reshape(-1, 1),
    training['publisher_encoded'].values.reshape(-1, 1),
    training["supporting_evidence"].values.reshape(-1, 1),
    training["vagueness_score"].values.reshape(-1, 1),
    training_title['information_gap_score'].values.reshape(-1, 1)
))
X_test = np.hstack((
    testing["compound"].values.reshape(-1, 1),
    testing["amplification_label"].values.reshape(-1, 1),
    testing["scarcity_label"].values.reshape(-1, 1),
    testing["info_complexity"].values.reshape(-1, 1),
    X_test_consistency.reshape(-1, 1),
    testing["coherence_score"].values.reshape(-1, 1),
    testing['publisher_encoded'].values.reshape(-1, 1),
    testing["supporting_evidence"].values.reshape(-1, 1),
    testing["vagueness_score"].values.reshape(-1, 1),
    testing_title['information_gap_score'].values.reshape(-1, 1)
))

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance Analysis
import textwrap
from matplotlib.ticker import PercentFormatter

feature_names = np.array([
    "Negativity Bias",
    "Confidence Heuristic",
    "Scarcity Principle",
    "Information Complexity",
    "Logical Consistency",
    "Coherence",
    "Source Credibility",
    "Supporting Evidence",
    "Language Vagueness",
    "Information Gap"
])

importances = np.array(model.feature_importances_, dtype=float)

order = np.argsort(importances)[::-1]
feat = feature_names[order]
imp = importances[order]

wrap = lambda s: "\n".join(textwrap.wrap(s, width=20, break_long_words=False))
feat_wrapped = [wrap(s) for s in feat]

total = imp.sum()
imp_pct = imp / total if total > 0 else imp

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars = ax.barh(feat_wrapped, imp_pct)
ax.invert_yaxis()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
ax.set_axisbelow(True)

ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.set_xlabel("Relative Importance (%)")
ax.set_title("Feature Importance (Ranked)")

for rect, val in zip(bars, imp_pct):
    x = rect.get_width()
    ax.text(
        x + max(0.01, 0.02 * imp_pct.max()),
        rect.get_y() + rect.get_height()/2,
        f"{val*100:.1f}%",
        va="center", ha="left"
    )

fig.tight_layout(rect=[0, 0, 0.98, 1])

plt.savefig("feature_importance.pdf", bbox_inches="tight")
plt.savefig("feature_importance.svg", bbox_inches="tight")
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

# Theory Importance Ranking
feature_importances = model.feature_importances_

theory_importance = {
    "Theory 1 (negativity_bias)": feature_importances[0],
    "Theory 2 (amplification_label)": feature_importances[1],
    "Theory 3 (scarcity_label)": feature_importances[2],
    "Theory 4 (info_complexity)": feature_importances[3],
    "Theory 5 (consistency_score)": feature_importances[4],
    "Theory 6 (coherence_score)": feature_importances[5],  
    "Theory 7 (credibility)": feature_importances[6],
    "Theory 8 (supporting_evidence)": feature_importances[7],
    "Theory 9 (language vagueness)": feature_importances[8],
    "Theory 10 (information gap)": feature_importances[9],
}

sorted_theories = sorted(theory_importance.items(), key=lambda x: -x[1])

print("Theory Importance Ranking:")
for theory, score in sorted_theories:
    print(f"{theory}: {score:.4f}")

# Correlation Heatmap
feature_names_list = [
    "Negativity Bias",
    "Confidence Heuristic",
    "Scarcity Principle",
    "Information Complexity",
    "Logical Consistency",
    "Coherence",
    "Source Credibility",
    "Supporting Evidence",
    "Language Vagueness",
    "Information Gap"
]

X_train_df = pd.DataFrame(X_train, columns=feature_names_list)
corr_matrix = X_train_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# SHAP Analysis
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

plt.figure()
shap.summary_plot(shap_values, features=X_train, feature_names=feature_names_list, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print("All analyses complete. Figures saved.")
