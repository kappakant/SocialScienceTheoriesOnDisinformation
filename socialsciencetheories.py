import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Dataset Assumptions:
# Article text in 'body_text'
# Article title in 'title'
# Article label (0 or 1) in 'label' 

# Load the datasets
datasetTrain = "~/csvs/FnRtrain.csv"
datasetTest = "~/csvs/FnRtest.csv"

training = pd.read_csv(datasetTrain)
testing  = pd.read_csv(datasetTest)

training["body_text"] = training["body_text"].str.lower()
testing["body_text"]  = testing["body_text"].str.lower()

X_train = training["body_text"]
y_train = training['label']
X_test = testing["body_text"]
y_test = testing['label']

# Theory 5 feature: Internal Validation

def detect_logical_consistency(texts):

    entailment_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=0)

    consistency_scores = []
    for text in tqdm(texts, desc="Processing Logical Consistency", unit="body_text"):
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
    plt.savefig("FNRlogic_confmatrix.pdf")

    return accuracy



print("Extracting logical consistency features for training set...")
X_train_consistency = detect_logical_consistency(X_train)
print("Extracting logical consistency features for testing set...")
X_test_consistency = detect_logical_consistency(X_test)


accuracy = train_and_evaluate(X_train_consistency, X_test_consistency, y_train, y_test)
print(f'Model Accuracy: {accuracy:.4f}')

# # Theory 6 feature: Coherence

import spacy
from collections import defaultdict, Counter
import math


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

testing['coherence_score'] = testing["body_text"].apply(compute_entity_grid_coherence)
training['coherence_score'] = training["body_text"].apply(compute_entity_grid_coherence)

# # theory 7 features: Credibility Source

from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Initialize LLM
LLM_model_name = "google/gemma-3-27b-it" 
LLM_tokenizer = AutoTokenizer.from_pretrained(LLM_model_name)
LLM_model = AutoModelForCausalLM.from_pretrained(
    LLM_model_name, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
device = next(LLM_model.parameters()).device

def get_credibility(text):
    truncated_text = text[:6000]
    
    # Prepare the prompt
    prompt = f""""Please help me evaluate the credibility of cited sources in the following text by checking the named researchers, institutions, publication venues, and their expertise/reputation. 
    Only respond with a single number: 0 if all sources appear credible, or 1 if any seem questionable. 
    Here's the text:{truncated_text}
    """
    
    # Tokenize and generate response
    input_ids = LLM_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = LLM_model.generate(**input_ids, max_new_tokens=10)
    response = LLM_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the score - look for the last number in the response
    score = None
    for token in reversed(response.split()):
        if token in ['0', '1']:
            score = token
            break
    
    return int(score)


print("Processing training data...")
training["credibility"] = [get_credibility(text) for text in tqdm(training["body_text"])]

print("\nProcessing test data...")
testing["credibility"] = [get_credibility(text) for text in tqdm(testing["body_text"])]

# theory 3 principle: Scarcity Bias

def get_scarcity_label(text):
    truncated_text = text[:6000]
    
    # Prompt based on the Scarcity Bias definition you provided
    prompt = f"""Scarcity bias means a message becomes more persuasive when it emphasizes limited availability, urgency, or time-sensitivity. 
For example: "Only today", "Hurry up before it's gone", or "This info may be deleted soon".

Determine whether the following news text uses scarcity bias tactics and respond with ONLY one number:
0 (does NOT use scarcity bias), 1 (uses scarcity bias).

News: {truncated_text}

Answer:"""

    # Tokenize and generate response
    input_ids = LLM_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = LLM_model.generate(**input_ids, max_new_tokens=10)
    response = LLM_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the label from the model output
    score = None
    for token in reversed(response.split()):
        if token in ['0', '1']:
            score = token
            break
    
    return int(score)

# Apply to training data
print("Processing training data for scarcity bias...")
training["scarcity_label"] = [get_scarcity_label(text) for text in tqdm(training["body_text"])]

# Apply to testing data
print("\nProcessing test data for scarcity bias...")
testing["scarcity_label"] = [get_scarcity_label(text) for text in tqdm(testing["body_text"])]

# theory 8 feature: Supporting Evidence

def get_supporting_evidence(text):
    truncated_text = text[:6000]
    
    # Prepare the prompt
    prompt = f"""Determine whether the following text contains supporting evidence and respond with ONLY one number: 
    0 (does not contain supporting evidence), 1 (contain supporting evidence). 

    News: {truncated_text}
    
    Answer:"""
    
    # Tokenize and generate response
    input_ids = LLM_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = LLM_model.generate(**input_ids, max_new_tokens=10)
    response = LLM_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the score - look for the last number in the response
    score = None
    for token in reversed(response.split()):
        if token in ['0', '1']:
            score = token
            break
    
    return int(score)


print("Processing training data...")
training["supporting_evidence"] = [get_supporting_evidence(text) for text in tqdm(training["body_text"])]

print("\nProcessing test data...")
testing["supporting_evidence"] = [get_supporting_evidence(text) for text in tqdm(testing["body_text"])]

# theory 10 features: Information Gap

# FNR dataset does not have puctuations in title at last EXCEPT for ?
training['title']= training['title'].str.lower()
testing['title']= testing['title'].str.lower()
training.head(2)

# Part 1: Punctuation
import re

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
    
# Part 2: Suprisingness
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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
    sscore = float(dkl / (1.0 + dkl))  #
    return sscore, dkl


# ================================= train =================================
print("Scoring training titles ...")
titles = training['title'].fillna("").astype(str).tolist()
vectorizer, lda, Q = fit_lda_and_baseline(titles, n_topics=20)

w_punct, w_surp = 0.5, 0.5 
punct_scores, surp_scores, kls = [], [], []

for title in tqdm(training['title'].fillna("").astype(str).tolist(), desc="Scoring titles"):
    pscore = punctuation_score(title)                  
    sscore, dkl = surprisingness_score(title, vectorizer, lda, Q)  
    punct_scores.append(pscore)
    surp_scores.append(sscore)
    kls.append(dkl)

training['punctuation_score'] = punct_scores
training['surprisingness_score'] = surp_scores
training['KL'] = kls  
training['information_gap_score'] = w_punct * training['punctuation_score'] + w_surp * training['surprisingness_score']

# ================================= test =================================
print("Scoring testing titles ...")
titles = testing['title'].fillna("").astype(str).tolist()
vectorizer, lda, Q = fit_lda_and_baseline(titles, n_topics=20)
w_punct, w_surp = 0.5, 0.5 
punct_scores, surp_scores, kls = [], [], []
for title in tqdm(testing['title'].fillna("").astype(str).tolist(), desc="Scoring titles"):
    pscore = punctuation_score(title)                   
    sscore, dkl = surprisingness_score(title, vectorizer, lda, Q)
    punct_scores.append(pscore)
    surp_scores.append(sscore)
    kls.append(dkl)
testing['punctuation_score'] = punct_scores
testing['surprisingness_score'] = surp_scores
testing['KL'] = kls 
testing['information_gap_score'] = w_punct * testing['punctuation_score'] + w_surp * testing['surprisingness_score']

del LLM_model
del LLM_tokenizer
torch.cuda.empty_cache()
gc.collect()

# vagueness score
df = pd.read_csv("~/csvs/vague_phrases_scores.csv")
# 'vague_phrases'（text）and 'scores'（1–5 Likert）
assert {'vague_phrases','scores'}.issubset(df.columns), "unmatched, check your column names"

# nomralized scores to [0,1]
df['score_01'] = (df['scores'] - df['scores'].min()) / (df['scores'].max() - df['scores'].min())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df[['vague_phrases', 'score_01']].rename(columns={'vague_phrases':"body_text",'score_01':'labels'}))
test_dataset  = Dataset.from_pandas(test_df[['vague_phrases', 'score_01']].rename(columns={'vague_phrases':"body_text",'score_01':'labels'}))

Vague_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
Vague_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=1,
    problem_type="regression"
).to(device)

def preprocess(batch):
    enc = Vague_tokenizer(batch["body_text"], truncation=True, padding='max_length', max_length=128)
    enc['labels'] = batch['labels'] 
    return enc

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
test_dataset  = test_dataset.map(preprocess,  batched=True, remove_columns=test_dataset.column_names)

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
training['vagueness_score'] = training["body_text"].progress_apply(get_vagueness_score)
testing['vagueness_score']  = testing["body_text"].progress_apply(get_vagueness_score)

print("Vagueness completed successfully, cleaning cache now")
del Vague_model
del Vague_tokenizer
torch.cuda.empty_cache()
gc.collect()

# Theory 1 Feature: Negativity Bias

# Initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get compound value
def get_compound(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Calculate compound values for training and testing sets
training['compound'] = training["body_text"].apply(get_compound)
testing['compound'] = testing["body_text"].apply(get_compound)



# Theory 2 Feature: Amplification Hypothesis

import re

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

training['amplification_label'] = training["body_text"].apply(classify_certainty)
testing['amplification_label'] = testing["body_text"].apply(classify_certainty)


import textstat

def calculate_text_complexity(text):
    return textstat.flesch_reading_ease(text)

training['info_complexity'] = training["body_text"].apply(calculate_text_complexity)
testing['info_complexity'] = testing["body_text"].apply(calculate_text_complexity)


# FINAL FND

X_train = np.hstack((
    training["compound"].values.reshape(-1, 1),            # theory 1
    training["amplification_label"].values.reshape(-1, 1), # theory 2
    training["scarcity_label"].values.reshape(-1, 1),      # theory 3
    training["info_complexity"].values.reshape(-1, 1),     # theory 4
    X_train_consistency.reshape(-1, 1),                    # theory 5 
    training["coherence_score"].values.reshape(-1, 1),     # theory 6
    training["credibility"].values.reshape(-1, 1),       # theory 7
    training["supporting_evidence"].values.reshape(-1, 1), # theory 8 
    training["vagueness_score"].values.reshape(-1, 1),     # theory 9
    training['information_gap_score'].values.reshape(-1, 1) # theory 10   # theory 10 
))
X_test = np.hstack((
    testing["compound"].values.reshape(-1, 1),             # theory 1
    testing["amplification_label"].values.reshape(-1, 1),  # theory 2
    testing["scarcity_label"].values.reshape(-1, 1),       # theory 3
    testing["info_complexity"].values.reshape(-1, 1),      # theory 4
    X_test_consistency.reshape(-1, 1),                     # theory 5             
    testing["coherence_score"].values.reshape(-1, 1),      # theory 6
    testing["credibility"].values.reshape(-1, 1),        # theory 7
    testing["supporting_evidence"].values.reshape(-1, 1),  # theory 8
    testing["vagueness_score"].values.reshape(-1, 1),      # theory 9
    testing['information_gap_score'].values.reshape(-1, 1) # theory 10   # theory 10
))



model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

feature_names = (
    ["Negativity Bias"] +       #theory 1
    ["Amplification Hypothesis"] +   #theory 2
    ["Scarcity Principle"] +        #theory 3
    ["Information Complexity"] +       #theory 4
    ["Infernal Validation"] +     #theory 5
    ["Coherence"] +       #theory 6
    ["Credibility Source"]+            #theory 7
    ["Supporting Evidence"] +   #theory 8
    ["Language Vagueness"] +                #theory 9
    ["Information Gap"]                  #theory 10   
)

importances = model.feature_importances_


importances = np.array(importances)
feature_names = np.array(feature_names)


sorted_idx = np.argsort(importances)[::-1]
sorted_features = feature_names[sorted_idx]
sorted_importances = importances[sorted_idx]


print("Feature Importance (sorted):")
for name, score in zip(sorted_features, sorted_importances):
    print(f"{name}: {score:.4f}")


plt.figure(figsize=(6, 4))
plt.barh(sorted_features, sorted_importances, color='lightgreen')
plt.xlabel("Feature Importance")
plt.title("Feature Importance (Sorted)")
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()

from matplotlib.ticker import PercentFormatter
import textwrap

# === Inputs ===
feature_names = np.array([
    "Negativity Bias",          # 1
    "Amplification Hypothesis", # 2
    "Scarcity Principle",       # 3
    "Information Complexity",   # 4
    "Infernal Validation",      # 5
    "Coherence",                # 6
    "Credibility Source",       # 7
    "Supporting Evidence",      # 8
    "Language Vagueness",       # 9
    "Information Gap"           # 10
])

importances = np.array(model.feature_importances_, dtype=float)

order = np.argsort(importances)[::-1]
feat = feature_names[order]
imp  = importances[order]

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

plt.savefig("FNRfeature_importance.pdf", bbox_inches="tight")
plt.savefig("FNRfeature_importance.svg", bbox_inches="tight")

plt.show()

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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


feature_names = (
    ["Negativity Bias"] +       #theory 1
    ["Amplification Hypothesis"] +   #theory 2
    ["Scarcity Principle"] +        #theory 3
    ["Information Complexity"] +       #theory 4
    ["Infernal Validation"] +     #theory 5
    ["Coherence"] +       #theory 6
    ["Credibility Source"]+            #theory 7
    ["Supporting Evidence"] +   #theory 8
    ["Language Vagueness"] +                #theory 9
    ["Information Gap"]                  #theory 10   
)

X_train_df = pd.DataFrame(X_train, columns=feature_names)


corr_matrix = X_train_df.corr()


plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import textwrap

feature_names = [
    "Negativity Bias",
    "Amplification Hypothesis",
    "Scarcity Principle",
    "Information Complexity",
    "Infernal Validation",
    "Coherence",
    "Credibility Source",
    "Supporting Evidence",
    "Language Vagueness",
    "Information Gap"
]

X = np.asarray(X_train, dtype=float)
assert X.shape[1] == len(feature_names), "列数与特征名数量不一致"

corr = np.corrcoef(X, rowvar=False) 
n = corr.shape[0]

eigvals, eigvecs = np.linalg.eigh(corr)
order = np.argsort(eigvecs[:, -1])
corr_ord = corr[np.ix_(order, order)]
feat_ord = [feature_names[i] for i in order]

wrap = lambda s: "\n".join(textwrap.wrap(s, width=18, break_long_words=False))
feat_wrapped = [wrap(s) for s in feat_ord]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

cell_inch = 0.6 
fig_w = cell_inch * n + 2.2  
fig_h = cell_inch * n + 1.8
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = ax.imshow(corr_ord, vmin=-1, vmax=1, aspect="equal")

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(feat_wrapped, rotation=45, ha="right")
ax.set_yticklabels(feat_wrapped)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", linestyle="-", linewidth=0.8, alpha=1.0)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r")

show_values = n <= 14
if show_values:
    for i in range(n):
        for j in range(n):
            val = corr_ord[i, j]
            norm = (val + 1) / 2
            txt_color = "black" if norm < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9,
                    fontweight="semibold" if abs(val) > 0.6 else "normal",
                    color=txt_color)

ax.set_title("Feature Correlation Heatmap")

fig.tight_layout()
plt.savefig("FNRcorrelation_heatmap.pdf", bbox_inches="tight")
plt.savefig("FNRcorrelation_heatmap.svg", bbox_inches="tight")
plt.show()

import shap


explainer = shap.Explainer(model, X_train)


shap_values = explainer(X_train)


shap.summary_plot(shap_values, features=X_train, feature_names=feature_names)
plt.savefig("FNRshap_value.pdf")





