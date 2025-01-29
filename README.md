# Shivaay LLM - GSM8K Benchmark Evaluation

This repository contains the evaluation code and results for the **GSM8K Benchmark** using the **Shivaay LLM** with **8-shot testing** and **Chain-of-Thought (CoT)** reasoning. The benchmark measures the model's ability to answer complex science questions.

Shivaay ranked 11th on PapersWithCode SOTA (State-of-the-Art) Leaderboard for GSM8K (Models not using extra dataset) scoring **87.41**

## Metrics

- **Evaluation Method**: 8-Shot Testing with Chain-of-Thought Reasoning
- **Primary Metric**: Accuracy (%) on the GSM8K Benchmark

## Prerequisites

- Python 3.6 or higher
- A valid **FuturixAI API Key** (obtain from [Shivaay Playground](https://shivaay.futurixai.com/playground))
- Required packages: `requests`, `python-dotenv`

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/FuturixAI-and-Quantum-Works/Shivaay_GSM8K
   cd Shivaay_GSM8K
   ```

2. **Set Up a Virtual Environment (recommended):**:

   ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/MacOS
    # OR
    venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies:**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Add API Key:**:

- Create a .env file in the root directory
- Add your FuturixAI API key:
  ```bash
  FUTURIXAI_API_KEY="your_api_key_here"
  ```

---

## Running the Evaluation

Execute the benchmark script:

```bash
python evaluate.py
```

### Output Files

1. `complete_responses.json`:
   Contains detailed model responses in the following format:

```json
[
  {
    "question": "A robe takes 2 bolts of blue fiber and half....",
    "answers": "3",
    "model_answers": "3",
    "model_completion": "A robe takes 2 bolts of blue fiber. It takes half...",
    "is_correct": true
  }
]
```

2. `scores.txt`:
   Provides final accuracy metrics:

```txt
Num of total question: 1319, Correct num: 1153, Accuracy: 0.8741470811220622.
```

Final Score: `Accuracy * 100` (e.g., 87.41% in our run).

---

### Notes

- **Invalid Responses**: 3 responses in the test set were tagged [invalid] due to parsing errors. These require manual evaluation.
- **Reproducibility**: To replicate our results, ensure you use the same 8-shot examples provided in the repository.
