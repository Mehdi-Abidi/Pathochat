# Evaluation on MedMCQA pathology questions

Added in November 2026. PathoChat had never been scored on a labelled set, so this folder measures it on exam questions with a public answer key and uses the result to change the model and the prompt.

## Questions

[MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) is a set of Indian postgraduate medical entrance exam questions with four options each. Its validation split has 259 single-answer questions tagged Pathology. A fixed random draw (`sample(n=100, random_state=0)`) takes 100 of them as the test set. The other 159 form the dev set.

The model was chosen on dev questions. The new prompt was written once, after early Mistral-7B runs showed the original prompt refusing or guessing when the passages lacked the answer, and it was not tuned against the test set. Each test run below was done once.

## Results on the 100 test questions

| Setup | Correct | Accuracy | 95% interval | No answer |
| --- | --- | --- | --- | --- |
| Mistral-7B, no retrieval | 60 of 100 | 60% | 50 to 69 | 0 |
| Mistral-7B with retrieval, original prompt (PathoChat as first built) | 59 of 100 | 59% | 49 to 68 | 4 |
| Qwen3.8-27B with retrieval, new prompt (PathoChat now) | 86 of 100 | 86% | 78 to 91 | 0 |
| Qwen3.8-27B, no retrieval | 91 of 100 | 91% | 84 to 95 | 0 |

"No answer" counts replies of "I don't know" or replies with no option letter. They are scored as wrong.

PathoChat went from 59 to 86 correct out of 100. On the 100 shared questions the new setup alone was right on 32 and the old setup alone on 5 (exact McNemar p < 0.0001).

## What changed

1. The prompt. The original prompt told the model to use only the retrieved context and to say "I don't know" otherwise. The passages often do not contain the answer: the text of the correct option appears in the top 5 passages for 22 of the 100 test questions. The new prompt tells the model to use the passages when they are relevant and standard pathology knowledge when they are not. With Mistral-7B this prompt scored 43 of 69 on the test questions it was run on, against 41 for the original prompt on the same questions (p = 0.62, not a reliable difference at this size).
2. The model. Two models on the Groq free tier were compared on 40 dev questions with retrieval and the new prompt: Qwen3.8-27B scored 38 of 40 and gpt-oss-120b 34 of 40. Qwen3.8-27B was chosen and then run once on the test set.

## Does retrieval help?

Not on these questions.

- Mistral-7B scored 60 without retrieval and 59 with it (p = 1.00). Retrieval made no measurable difference.
- Qwen3.8-27B scored 91 without retrieval and 86 with it. Retrieval never turned a wrong answer into a right one, and on 5 questions it turned a right answer into a wrong one (p = 0.062).

So on exam questions the improvement comes from the model and the prompt. The passages give the app its textbook citations, but they did not add accuracy for either model, and when they miss the point they can mislead a model that already knows the answer. Filtering out weak passages before they reach the model is the next thing to try.

## Limits

- 100 questions give wide intervals. Differences of a few points mean little.
- Multiple-choice exam questions are not the same as the open questions people type into the app.
- MedMCQA is public, so either model may have seen these questions in training.
- The Mistral-7B runs used a 4-bit copy of Mistral-7B-Instruct-v0.3 (Q4_K_M, llama.cpp) on a laptop at temperature 0 with 5 passages. The deployed app used the Hugging Face endpoint at temperature 0.5 with 4 passages.
- The grounded-prompt run with Mistral-7B covers 69 of the 100 test questions.

## Reproduce

```bash
pip install -r evaluation/requirements.txt

export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_API_KEY=your_key
python evaluation/evaluate.py run --name qwen38_tutor --model qwen/qwen3.8-27b --prompt tutor
python evaluation/evaluate.py run --name qwen38_tutor_no_retrieval --model qwen/qwen3.8-27b --prompt tutor_no_retrieval

python evaluation/evaluate.py report mistral7b_original qwen38_tutor
```

For the Mistral-7B runs, start a llama.cpp server with the model on port 8080 and leave `LLM_BASE_URL` unset. The prompts are `direct`, `original` and `grounded`.

Each file in `evaluation/results/` holds one line per question: the MedMCQA id, the correct option, the model's option, and whether it was right. The first run downloads the MedMCQA validation file (about 1.5 MB) into `evaluation/data/`.
