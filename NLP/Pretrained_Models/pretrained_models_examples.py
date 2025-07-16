"""
Pretrained Language Models Examples
==================================

- BERT embedding extraction (transformers)
- Sentence classification (BERT fine-tuning)
- GPT-2 text generation
- Sentence similarity, sentence embeddings
"""

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. BERT embedding extraction
print("\n=== BERT Embedding Extraction ===")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentences = [
    "Natural language processing is a fascinating field of artificial intelligence.",
    "Machine learning models can understand and generate human language.",
    "Deep learning has revolutionized the way we approach text analysis.",
    "Transformers architecture has become the foundation of modern NLP."
]

for sent in sentences:
    inputs = tokenizer(sent, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    print(f"Sentence: {sent}\n[CLS] embedding (first 5 dims): {cls_embedding[:5]}")

# 2. BERT sentence classification (pipeline)
print("\n=== BERT Sentiment Classification (pipeline) ===")
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
results = classifier([
    "This product exceeded all my expectations!",
    "The service was terrible and the staff was rude.",
    "The movie was okay, nothing special but not bad either.",
    "I absolutely love this new technology!"
])
for r in results:
    print(r)

# 3. GPT-2 text generation
print("\n=== GPT-2 Text Generation ===")
try:
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Add padding token
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    prompt = "The future of artificial intelligence"
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs, 
            max_length=50, 
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
except Exception as e:
    print("GPT-2 generation failed:", e)

# 4. Sentence similarity (BERT embedding cosine similarity)
print("\n=== Sentence Similarity (BERT Embedding) ===")
embeddings = []
for sent in sentences:
    inputs = tokenizer(sent, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    embeddings.append(cls_embedding)

# Calculate pairwise similarities
print("Sentence similarity matrix:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        print(f"Sentences {i+1} & {j+1}: {sim:.4f}")

# 5. Named Entity Recognition with BERT
print("\n=== Named Entity Recognition ===")
ner_pipeline = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')
ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
ner_results = ner_pipeline(ner_text)
print(f"Text: {ner_text}")
for entity in ner_results:
    print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.3f}")

# 6. Question Answering with BERT
print("\n=== Question Answering ===")
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
context = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions."
question = "What is machine learning?"
qa_result = qa_pipeline(question=question, context=context)
print(f"Question: {question}")
print(f"Answer: {qa_result['answer']}")
print(f"Confidence: {qa_result['score']:.3f}")

# 7. Text summarization
print("\n=== Text Summarization ===")
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
long_text = """
Artificial intelligence has transformed numerous industries in recent years. From healthcare to finance, 
AI systems are being deployed to automate tasks, improve decision-making, and enhance user experiences. 
Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions 
with remarkable accuracy. Deep learning models, particularly neural networks, have achieved breakthrough 
results in image recognition, natural language processing, and speech synthesis. The development of 
large language models like GPT and BERT has revolutionized how we interact with text data, enabling 
more sophisticated chatbots, translation services, and content generation tools.
"""
summary = summarizer(long_text, max_length=100, min_length=30, do_sample=False)
print(f"Original text length: {len(long_text)} characters")
print(f"Summary: {summary[0]['summary_text']}")
print(f"Summary length: {len(summary[0]['summary_text'])} characters") 